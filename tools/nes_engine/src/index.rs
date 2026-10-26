//! Workspace symbol index.
//!
//! A parallel, gitignore-aware walk parses source files with tree-sitter tags
//! queries (the same definitions GitHub code navigation and aider's repo map
//! use) and records every definition with a short snippet. Rescans reuse the
//! parse of any file whose mtime and size are unchanged, so after the first
//! build a refresh is a parallel stat walk plus a few reparses.

use anyhow::{Context as _, Result};
use ignore::WalkBuilder;
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Instant, SystemTime};
use tree_sitter::{Language, Parser, Query, QueryCursor, StreamingIterator};

pub const MAX_FILE_BYTES: u64 = 512 * 1024;
pub const MAX_FILES: usize = 40_000;
pub const SNIPPET_LINES: usize = 25;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum Lang {
    Rust,
    Python,
    TypeScript,
    Tsx,
    JavaScript,
    Go,
    C,
    Cpp,
    Java,
    Lua,
    Nix,
}

impl Lang {
    pub fn from_path(path: &Path) -> Option<Lang> {
        let ext = path.extension()?.to_str()?.to_ascii_lowercase();
        Some(match ext.as_str() {
            "rs" => Lang::Rust,
            "py" | "pyi" => Lang::Python,
            "ts" | "mts" | "cts" => Lang::TypeScript,
            "tsx" => Lang::Tsx,
            "js" | "jsx" | "mjs" | "cjs" => Lang::JavaScript,
            "go" => Lang::Go,
            "c" | "h" => Lang::C,
            "cc" | "cpp" | "cxx" | "hpp" | "hh" | "hxx" => Lang::Cpp,
            "java" => Lang::Java,
            "lua" => Lang::Lua,
            "nix" => Lang::Nix,
            _ => return None,
        })
    }

    fn language(self) -> Language {
        match self {
            Lang::Rust => tree_sitter_rust::LANGUAGE.into(),
            Lang::Python => tree_sitter_python::LANGUAGE.into(),
            Lang::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            Lang::Tsx => tree_sitter_typescript::LANGUAGE_TSX.into(),
            Lang::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
            Lang::Go => tree_sitter_go::LANGUAGE.into(),
            Lang::C => tree_sitter_c::LANGUAGE.into(),
            Lang::Cpp => tree_sitter_cpp::LANGUAGE.into(),
            Lang::Java => tree_sitter_java::LANGUAGE.into(),
            Lang::Lua => tree_sitter_lua::LANGUAGE.into(),
            Lang::Nix => tree_sitter_nix::LANGUAGE.into(),
        }
    }

    fn query_sources(self) -> Vec<&'static str> {
        match self {
            Lang::Rust => vec![tree_sitter_rust::TAGS_QUERY, RUST_EXTRA],
            Lang::Python => vec![tree_sitter_python::TAGS_QUERY],
            // The TypeScript grammar extends JavaScript; combine both tag sets.
            Lang::TypeScript | Lang::Tsx => {
                vec![tree_sitter_javascript::TAGS_QUERY, tree_sitter_typescript::TAGS_QUERY, TS_EXTRA]
            }
            Lang::JavaScript => vec![tree_sitter_javascript::TAGS_QUERY],
            Lang::Go => vec![tree_sitter_go::TAGS_QUERY],
            Lang::C => vec![tree_sitter_c::TAGS_QUERY],
            Lang::Cpp => vec![tree_sitter_cpp::TAGS_QUERY],
            Lang::Java => vec![tree_sitter_java::TAGS_QUERY],
            Lang::Lua => vec![tree_sitter_lua::TAGS_QUERY],
            Lang::Nix => vec![NIX_DEFS],
        }
    }
}

const RUST_EXTRA: &str = r#"
(const_item name: (identifier) @name) @definition.constant
(static_item name: (identifier) @name) @definition.constant
(macro_definition name: (identifier) @name) @definition.macro
(impl_item type: (type_identifier) @name) @definition.implementation
"#;

const TS_EXTRA: &str = r#"
(type_alias_declaration name: (type_identifier) @name) @definition.type
(enum_declaration name: (identifier) @name) @definition.enum
(lexical_declaration (variable_declarator name: (identifier) @name value: (arrow_function))) @definition.function
"#;

const NIX_DEFS: &str = r#"
(binding attrpath: (attrpath) @name expression: (function_expression)) @definition.function
(binding attrpath: (attrpath) @name expression: (attrset_expression)) @definition.module
"#;

struct Compiled {
    language: Language,
    query: Query,
    name_index: u32,
}

fn compiled(lang: Lang) -> Option<&'static Compiled> {
    static CACHE: OnceLock<Mutex<HashMap<Lang, Option<&'static Compiled>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().unwrap();
    *guard.entry(lang).or_insert_with(|| {
        let language = lang.language();
        // Try the combined query, then each part alone, so a grammar/query
        // version skew degrades to fewer definitions rather than none.
        let sources = lang.query_sources();
        let combined = sources.join("\n");
        let query = Query::new(&language, &combined).ok().or_else(|| {
            sources.iter().find_map(|s| Query::new(&language, s).ok())
        })?;
        let name_index = query.capture_index_for_name("name")?;
        Some(&*Box::leak(Box::new(Compiled { language, query, name_index })))
    })
}

#[derive(Clone, Debug, Serialize)]
pub struct Definition {
    pub name: Arc<str>,
    pub kind: Arc<str>,
    pub start_line: usize,
    pub end_line: usize,
    pub snippet: Arc<str>,
}

#[derive(Clone, Debug)]
pub struct FileEntry {
    pub path: PathBuf,
    pub rel: String,
    pub mtime: Option<SystemTime>,
    pub size: u64,
    pub defs: Arc<Vec<Definition>>,
}

/// Parse a file's definitions with tree-sitter tags queries.
pub fn parse_definitions(lang: Lang, source: &str) -> Vec<Definition> {
    let Some(c) = compiled(lang) else {
        return Vec::new();
    };
    thread_local! {
        static PARSERS: std::cell::RefCell<HashMap<Lang, Parser>> = std::cell::RefCell::new(HashMap::new());
    }
    let tree = PARSERS.with(|p| {
        let mut map = p.borrow_mut();
        let parser = map.entry(lang).or_insert_with(|| {
            let mut parser = Parser::new();
            parser.set_language(&c.language).expect("grammar ABI");
            parser
        });
        parser.parse(source, None)
    });
    let Some(tree) = tree else {
        return Vec::new();
    };
    let lines: Vec<&str> = source.lines().collect();
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&c.query, tree.root_node(), source.as_bytes());
    let mut out: Vec<Definition> = Vec::new();
    let names = c.query.capture_names();
    while let Some(m) = matches.next() {
        let mut name = None;
        let mut def = None;
        for cap in m.captures {
            if cap.index == c.name_index {
                name = Some(cap.node);
            } else if let Some(kind) = names[cap.index as usize].strip_prefix("definition.") {
                def = Some((cap.node, kind));
            }
        }
        let (Some(name), Some((node, kind))) = (name, def) else {
            continue;
        };
        let Ok(name_text) = name.utf8_text(source.as_bytes()) else {
            continue;
        };
        if name_text.is_empty() || name_text.len() > 128 {
            continue;
        }
        let start_line = node.start_position().row;
        let end_line = node.end_position().row;
        if out.iter().any(|d| d.start_line == start_line && &*d.name == name_text) {
            continue;
        }
        let snippet_end = (end_line + 1).min(start_line + SNIPPET_LINES).min(lines.len());
        let snippet = lines[start_line.min(lines.len())..snippet_end].join("\n");
        out.push(Definition {
            name: Arc::from(name_text),
            kind: Arc::from(kind),
            start_line,
            end_line,
            snippet: Arc::from(snippet),
        });
    }
    out
}

#[derive(Default, Clone, Debug, Serialize)]
pub struct IndexStats {
    pub files: usize,
    pub definitions: usize,
    pub reparsed: usize,
    pub build_ms: f64,
    pub generation: u64,
}

#[derive(Default)]
pub struct Workspace {
    pub root: PathBuf,
    pub files: HashMap<PathBuf, FileEntry>,
    pub by_name: HashMap<Arc<str>, Vec<(PathBuf, usize)>>,
    pub stats: IndexStats,
    pub built_at: Option<Instant>,
}

impl Workspace {
    /// Build or refresh from disk, reusing unchanged parses from `self`.
    pub fn refresh(&self, root: &Path) -> Result<Workspace> {
        let started = Instant::now();
        let mut walker = WalkBuilder::new(root);
        walker
            .hidden(true)
            .git_ignore(true)
            .git_global(true)
            .git_exclude(true)
            .parents(true)
            .follow_links(false)
            .threads(4)
            .filter_entry(|e| {
                let name = e.file_name().to_string_lossy();
                !matches!(name.as_ref(), "node_modules" | "target" | ".git" | "result" | "dist" | "build" | "__pycache__" | ".venv" | "venv")
            });
        let found: Mutex<Vec<(PathBuf, Option<SystemTime>, u64)>> = Mutex::new(Vec::new());
        walker.build_parallel().run(|| {
            Box::new(|entry| {
                let Ok(entry) = entry else {
                    return ignore::WalkState::Continue;
                };
                if !entry.file_type().is_some_and(|t| t.is_file()) {
                    return ignore::WalkState::Continue;
                }
                let path = entry.path();
                if Lang::from_path(path).is_none() {
                    return ignore::WalkState::Continue;
                }
                let Ok(meta) = entry.metadata() else {
                    return ignore::WalkState::Continue;
                };
                if meta.len() > MAX_FILE_BYTES {
                    return ignore::WalkState::Continue;
                }
                let mut f = found.lock().unwrap();
                if f.len() >= MAX_FILES {
                    return ignore::WalkState::Quit;
                }
                f.push((path.to_path_buf(), meta.modified().ok(), meta.len()));
                ignore::WalkState::Continue
            })
        });
        let found = found.into_inner().unwrap();
        let reparsed = std::sync::atomic::AtomicUsize::new(0);
        let entries: Vec<FileEntry> = found
            .par_iter()
            .filter_map(|(path, mtime, size)| {
                if let Some(old) = self.files.get(path) {
                    if old.mtime == *mtime && old.size == *size && mtime.is_some() {
                        return Some(old.clone());
                    }
                }
                let lang = Lang::from_path(path)?;
                let bytes = std::fs::read(path).ok()?;
                if bytes.contains(&0) {
                    return None;
                }
                let source = String::from_utf8(bytes).ok()?;
                reparsed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Some(FileEntry {
                    rel: rel_path(root, path),
                    path: path.clone(),
                    mtime: *mtime,
                    size: *size,
                    defs: Arc::new(parse_definitions(lang, &source)),
                })
            })
            .collect();
        let mut ws = Workspace {
            root: root.to_path_buf(),
            ..Default::default()
        };
        for e in entries {
            ws.files.insert(e.path.clone(), e);
        }
        ws.rebuild_names();
        ws.stats = IndexStats {
            files: ws.files.len(),
            definitions: ws.by_name.values().map(Vec::len).sum(),
            reparsed: reparsed.into_inner(),
            build_ms: started.elapsed().as_secs_f64() * 1000.0,
            generation: self.stats.generation + 1,
        };
        ws.built_at = Some(Instant::now());
        Ok(ws)
    }

    fn rebuild_names(&mut self) {
        self.by_name.clear();
        for (path, entry) in &self.files {
            for (i, d) in entry.defs.iter().enumerate() {
                self.by_name.entry(d.name.clone()).or_default().push((path.clone(), i));
            }
        }
    }

    /// Replace one file's definitions from an in-memory buffer.
    pub fn update_buffer(&mut self, path: &Path, source: &str) {
        let Some(lang) = Lang::from_path(path) else {
            return;
        };
        if !path.starts_with(&self.root) || source.len() as u64 > MAX_FILE_BYTES {
            return;
        }
        let defs = Arc::new(parse_definitions(lang, source));
        if let Some(old) = self.files.get(path) {
            for d in old.defs.iter() {
                if let Some(v) = self.by_name.get_mut(&d.name) {
                    v.retain(|(p, _)| p != path);
                }
            }
        }
        for (i, d) in defs.iter().enumerate() {
            self.by_name.entry(d.name.clone()).or_default().push((path.to_path_buf(), i));
        }
        let rel = rel_path(&self.root, path);
        // Keep the disk mtime so the next rescan does not discard the buffer
        // parse unless the file actually changes on disk.
        let (mtime, size) = std::fs::metadata(path)
            .map(|m| (m.modified().ok(), m.len()))
            .unwrap_or((None, 0));
        self.files.insert(path.to_path_buf(), FileEntry { path: path.to_path_buf(), rel, mtime, size, defs });
    }

    pub fn lookup(&self, name: &str) -> impl Iterator<Item = (&FileEntry, &Definition)> {
        self.by_name
            .get(name)
            .into_iter()
            .flatten()
            .filter_map(|(p, i)| {
                let f = self.files.get(p)?;
                Some((f, f.defs.get(*i)?))
            })
    }
}

pub fn rel_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Find the workspace root: nearest ancestor with a VCS or project marker.
pub fn find_root(path: &Path) -> Option<PathBuf> {
    let markers = [".git", ".hg", "Cargo.toml", "pyproject.toml", "package.json", "go.mod", "flake.nix"];
    let mut best: Option<PathBuf> = None;
    let mut dir = path.parent();
    while let Some(d) = dir {
        if d.join(".git").exists() || d.join(".hg").exists() {
            return Some(d.to_path_buf());
        }
        if best.is_none() && markers.iter().any(|m| d.join(m).exists()) {
            best = Some(d.to_path_buf());
        }
        dir = d.parent();
    }
    best
}

/// Thread-safe registry of workspaces with background refresh.
#[derive(Default)]
pub struct Registry {
    inner: Mutex<HashMap<PathBuf, Arc<Mutex<Workspace>>>>,
    refreshing: Mutex<std::collections::HashSet<PathBuf>>,
}

impl Registry {
    pub fn get(&self, root: &Path) -> Arc<Mutex<Workspace>> {
        let mut inner = self.inner.lock().unwrap();
        inner
            .entry(root.to_path_buf())
            .or_insert_with(|| {
                Arc::new(Mutex::new(Workspace {
                    root: root.to_path_buf(),
                    ..Default::default()
                }))
            })
            .clone()
    }

    /// Refresh in a background thread when stale. Never blocks a request.
    pub fn ensure_fresh(self: &Arc<Self>, root: &Path, max_age_secs: u64) {
        let ws = self.get(root);
        let stale = ws
            .lock()
            .unwrap()
            .built_at
            .is_none_or(|t| t.elapsed().as_secs() >= max_age_secs);
        if !stale {
            return;
        }
        if !self.refreshing.lock().unwrap().insert(root.to_path_buf()) {
            return;
        }
        let this = Arc::clone(self);
        let root = root.to_path_buf();
        std::thread::Builder::new()
            .name("nes-index".into())
            .spawn(move || {
                let result = (|| -> Result<()> {
                    let snapshot = {
                        let guard = ws.lock().unwrap();
                        Workspace {
                            root: guard.root.clone(),
                            files: guard.files.clone(),
                            stats: guard.stats.clone(),
                            ..Default::default()
                        }
                    };
                    let fresh = snapshot.refresh(&root).context("index refresh")?;
                    *ws.lock().unwrap() = fresh;
                    Ok(())
                })();
                if let Err(e) = result {
                    eprintln!("nes: index refresh failed for {}: {e:#}", root.display());
                }
                this.refreshing.lock().unwrap().remove(&root);
            })
            .ok();
    }

    pub fn stats(&self) -> Vec<(PathBuf, IndexStats)> {
        self.inner
            .lock()
            .unwrap()
            .iter()
            .map(|(p, w)| (p.clone(), w.lock().unwrap().stats.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_definitions() {
        let src = "struct Foo { a: i32 }\nimpl Foo {\n    fn bar(&self) -> i32 { self.a }\n}\nconst LIMIT: usize = 3;\nfn top() {}\n";
        let defs = parse_definitions(Lang::Rust, src);
        let names: Vec<&str> = defs.iter().map(|d| &*d.name).collect();
        for n in ["Foo", "bar", "LIMIT", "top"] {
            assert!(names.contains(&n), "{n} missing from {names:?}");
        }
    }

    #[test]
    fn python_and_ts_definitions() {
        let py = parse_definitions(Lang::Python, "class A:\n    def m(self):\n        pass\n\ndef f():\n    return 1\n");
        let names: Vec<&str> = py.iter().map(|d| &*d.name).collect();
        assert!(names.contains(&"A") && names.contains(&"m") && names.contains(&"f"));
        let ts = parse_definitions(Lang::TypeScript, "export function g(x: number) { return x }\ninterface I { a: string }\ntype T = number;\nconst h = (y: number) => y;\n");
        let names: Vec<&str> = ts.iter().map(|d| &*d.name).collect();
        for n in ["g", "I", "T", "h"] {
            assert!(names.contains(&n), "{n} missing from {names:?}");
        }
    }

    #[test]
    fn every_language_query_compiles() {
        for lang in [Lang::Rust, Lang::Python, Lang::TypeScript, Lang::Tsx, Lang::JavaScript, Lang::Go, Lang::C, Lang::Cpp, Lang::Java, Lang::Lua, Lang::Nix] {
            assert!(compiled(lang).is_some(), "{lang:?} query failed");
        }
    }

    #[test]
    fn refresh_reuses_unchanged_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), "fn alpha() {}\n").unwrap();
        std::fs::write(dir.path().join("b.py"), "def beta():\n    pass\n").unwrap();
        let ws = Workspace::default().refresh(dir.path()).unwrap();
        assert_eq!(ws.stats.files, 2);
        assert_eq!(ws.stats.reparsed, 2);
        assert_eq!(ws.lookup("alpha").count(), 1);
        let again = ws.refresh(dir.path()).unwrap();
        assert_eq!(again.stats.reparsed, 0);
        assert_eq!(again.lookup("beta").count(), 1);
    }
}
