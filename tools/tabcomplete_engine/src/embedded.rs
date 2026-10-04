//! A seekable, mmap-compatible GGUF appended to the executable, never extracted.
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    ffi::CStr,
    fs::File,
    io::{Read, Seek, SeekFrom},
    ptr::NonNull,
};

const MAGIC: &[u8; 16] = b"TABCOMPLETEGGUF1";
const TRAILER_BYTES: u64 = 20;
const MAX_METADATA: u32 = 2 * 1024 * 1024;
const MAX_PAYLOAD: u64 = 1024 * 1024 * 1024;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Payload {
    pub version: u32,
    pub offset: u64,
    pub length: u64,
    pub sha256: String,
    pub alias: String,
    pub protocol: String,
    pub output_tokens: usize,
    pub context_size: u32,
    pub input_tokens: usize,
    pub batch_size: u32,
    pub microbatch_size: u32,
    pub threads: i32,
    pub cache_type: String,
    pub context_layout: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fim_profile: Option<crate::fim_v1::ServingProfile>,
}

impl Payload {
    pub fn inspect_executable() -> Result<Self> {
        Self::read(&mut File::open("/proc/self/exe")?)
    }

    fn read(file: &mut (impl Read + Seek)) -> Result<Self> {
        let size = file.seek(SeekFrom::End(0))?;
        ensure!(size >= TRAILER_BYTES, "embedded_footer_missing");
        file.seek(SeekFrom::End(-(TRAILER_BYTES as i64)))?;
        let mut length = [0; 4];
        let mut magic = [0; 16];
        file.read_exact(&mut length)?;
        file.read_exact(&mut magic)?;
        ensure!(&magic == MAGIC, "embedded_footer_missing");
        let length = u32::from_le_bytes(length);
        ensure!(
            (1..=MAX_METADATA).contains(&length),
            "embedded_metadata_bounds"
        );
        let start = size
            .checked_sub(TRAILER_BYTES + u64::from(length))
            .ok_or_else(|| anyhow::anyhow!("embedded_metadata_bounds"))?;
        file.seek(SeekFrom::Start(start))?;
        let mut metadata = vec![0; length as usize];
        file.read_exact(&mut metadata)?;
        let payload: Self = serde_json::from_slice(&metadata)?;
        ensure!(payload.version == 1, "embedded_version");
        ensure!(
            payload.offset >= 4096
                && payload.offset <= 64 * 1024 * 1024
                && payload.offset.is_multiple_of(4096),
            "embedded_offset_bounds"
        );
        ensure!(
            (4..=MAX_PAYLOAD).contains(&payload.length)
                && payload.offset.checked_add(payload.length) == Some(start),
            "embedded_payload_bounds"
        );
        ensure!(
            payload.sha256.len() == 64
                && payload
                    .sha256
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "embedded_digest_format"
        );
        let research_fim = matches!(
            (
                payload.alias.as_str(),
                payload.protocol.as_str(),
                payload.output_tokens
            ),
            (
                "q25-fim",
                crate::fim_v1::WIRE_VERSION,
                crate::fim_v1::OUTPUT_TOKEN_CAP
            )
        );
        ensure!(
            research_fim
                || matches!(
                    (
                        payload.alias.as_str(),
                        payload.protocol.as_str(),
                        payload.output_tokens
                    ),
                    ("q25", "single-line-edit-v1", 64) | ("sweep", "sweep-full-file-v1", 192)
                ),
            "embedded_model_profile"
        );
        ensure!(
            if research_fim {
                payload.fim_profile.is_some()
                    && payload.context_layout == crate::fim_v1::CONTEXT_LAYOUT
            } else {
                payload.fim_profile.is_none()
            },
            "embedded_fim_profile"
        );
        if let Some(profile) = &payload.fim_profile {
            crate::fim_v1::validate_serving_profile(profile)?;
            ensure!(
                profile
                    .tokenizer
                    .special_tokens
                    .windows(2)
                    .all(|pair| pair[0].id < pair[1].id),
                "embedded_fim_control_token_order"
            );
        }
        ensure!(
            payload.context_size == 2304
                && payload.input_tokens == 1024
                && payload.microbatch_size == 64
                && payload.batch_size == 256
                && payload.threads == 4
                && payload.cache_type == "f16"
                && (if research_fim {
                    payload.context_layout == crate::fim_v1::CONTEXT_LAYOUT
                } else {
                    payload.context_layout == "cursor-last-v1"
                }),
            "embedded_defaults"
        );
        file.seek(SeekFrom::Start(0))?;
        let mut elf = [0; 4];
        file.read_exact(&mut elf)?;
        ensure!(&elf == b"\x7fELF", "embedded_executable_format");
        file.seek(SeekFrom::Start(payload.offset))?;
        let mut gguf = [0; 4];
        file.read_exact(&mut gguf)?;
        ensure!(&gguf == b"GGUF", "embedded_gguf_header");
        payload.verify_reader(file)?;
        Ok(payload)
    }

    fn verify_reader(&self, file: &mut (impl Read + Seek)) -> Result<()> {
        file.seek(SeekFrom::Start(self.offset))?;
        let mut hash = Sha256::new();
        let mut remaining = self.length;
        let mut buffer = [0; 65536];
        while remaining > 0 {
            let count = remaining.min(buffer.len() as u64) as usize;
            file.read_exact(&mut buffer[..count])?;
            hash.update(&buffer[..count]);
            remaining -= count as u64;
        }
        ensure!(
            format!("{:x}", hash.finalize()) == self.sha256,
            "embedded_model_identity_mismatch"
        );
        Ok(())
    }

    pub fn open_stream(&self) -> Result<ModelFile> {
        // fopen opens exactly this process's immutable executable. fseeko leaves
        // it at the aligned GGUF header; the loader maps the same inode.
        self.open_stream_at(c"/proc/self/exe")
    }

    fn open_stream_at(&self, executable: &CStr) -> Result<ModelFile> {
        let pointer = unsafe { libc::fopen(executable.as_ptr(), c"rb".as_ptr()) };
        let file =
            ModelFile(NonNull::new(pointer).ok_or_else(|| anyhow::anyhow!("embedded_file_open"))?);
        ensure!(
            unsafe { libc::fseeko(file.0.as_ptr(), self.offset as libc::off_t, libc::SEEK_SET) }
                == 0,
            "embedded_file_seek"
        );
        Ok(file)
    }
}

pub struct ModelFile(NonNull<libc::FILE>);
impl ModelFile {
    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.0.as_ptr().cast()
    }
}
impl Drop for ModelFile {
    fn drop(&mut self) {
        unsafe {
            libc::fclose(self.0.as_ptr());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    fn profile() -> Payload {
        Payload {
            version: 1,
            offset: 4096,
            length: 8,
            sha256: format!("{:x}", Sha256::digest(b"GGUFtest")),
            alias: "q25".into(),
            protocol: "single-line-edit-v1".into(),
            output_tokens: 64,
            context_size: 2304,
            input_tokens: 1024,
            batch_size: 256,
            microbatch_size: 64,
            threads: 4,
            cache_type: "f16".into(),
            context_layout: "cursor-last-v1".into(),
            fim_profile: None,
        }
    }
    fn serving_profile() -> crate::fim_v1::ServingProfile {
        let tokenizer_vocab_ids: Vec<i32> = (0..=crate::fim_v1::FIM_SUFFIX_TOKEN_ID).collect();
        let tokenizer_vocab_ids_sha256 = format!(
            "{:x}",
            Sha256::digest(
                tokenizer_vocab_ids
                    .iter()
                    .map(|id| format!("{id}\n"))
                    .collect::<String>()
                    .as_bytes()
            )
        );
        let mut tokenizer = crate::fim_v1::TokenizerProfile {
            tokenizer_id: "synthetic/fim-fixture".into(),
            tokenizer_revision: "synthetic-revision".into(),
            tokenizer_sha256: "a".repeat(64),
            tokenizer_contract_sha256: String::new(),
            tokenizer_vocab_size: tokenizer_vocab_ids.len(),
            tokenizer_vocab_ids_sha256,
            tokenizer_vocab_ids,
            eos_id: crate::fim_v1::EOS_TOKEN_ID,
            fim_prefix_id: crate::fim_v1::FIM_PREFIX_TOKEN_ID,
            fim_suffix_id: crate::fim_v1::FIM_SUFFIX_TOKEN_ID,
            fim_middle_id: crate::fim_v1::FIM_MIDDLE_TOKEN_ID,
            completion_mode: crate::fim_v1::COMPLETION_MODE.into(),
            special_tokens: vec![
                crate::fim_v1::SpecialToken {
                    id: crate::fim_v1::EOS_TOKEN_ID,
                    spelling: crate::fim_v1::EOS_SPELLING.into(),
                },
                crate::fim_v1::SpecialToken {
                    id: crate::fim_v1::FIM_PREFIX_TOKEN_ID,
                    spelling: crate::fim_v1::FIM_PREFIX.into(),
                },
                crate::fim_v1::SpecialToken {
                    id: crate::fim_v1::FIM_MIDDLE_TOKEN_ID,
                    spelling: crate::fim_v1::FIM_MIDDLE.into(),
                },
                crate::fim_v1::SpecialToken {
                    id: crate::fim_v1::FIM_SUFFIX_TOKEN_ID,
                    spelling: crate::fim_v1::FIM_SUFFIX.into(),
                },
            ],
        };
        tokenizer.tokenizer_contract_sha256 = format!(
            "{:x}",
            Sha256::digest(crate::fim_v1::tokenizer_contract_bytes(&tokenizer).unwrap())
        );
        crate::fim_v1::ServingProfile {
            artifact_manifest_sha256: "b".repeat(64),
            tokenizer,
        }
    }
    fn container(p: &Payload) -> Vec<u8> {
        let mut bytes = vec![0; 4096];
        bytes[..4].copy_from_slice(b"\x7fELF");
        bytes.extend(b"GGUFtest");
        let mut metadata_value = serde_json::to_value(p).unwrap();
        if let Some(profile) = &p.fim_profile {
            metadata_value["fim_profile"]["tokenizer"]["tokenizer_vocab_ids"] =
                serde_json::json!(profile.tokenizer.tokenizer_vocab_ids);
        }
        let metadata = serde_json::to_vec(&metadata_value).unwrap();
        bytes.extend(&metadata);
        bytes.extend((metadata.len() as u32).to_le_bytes());
        bytes.extend(MAGIC);
        bytes
    }
    #[test]
    fn embedded_profiles_and_defaults() {
        for (alias, protocol, cap) in [
            ("q25", "single-line-edit-v1", 64),
            ("sweep", "sweep-full-file-v1", 192),
        ] {
            let mut p = profile();
            p.alias = alias.into();
            p.protocol = protocol.into();
            p.output_tokens = cap;
            let parsed = Payload::read(&mut Cursor::new(container(&p))).unwrap();
            assert_eq!(parsed.alias, alias);
            assert_eq!(parsed.output_tokens, cap);
            assert!(parsed.fim_profile.is_none());
        }
    }
    #[test]
    fn accepts_full_fim_profile_and_vocabulary_inventory() {
        let mut p = profile();
        p.alias = "q25-fim".into();
        p.protocol = crate::fim_v1::WIRE_VERSION.into();
        p.output_tokens = crate::fim_v1::OUTPUT_TOKEN_CAP;
        p.context_layout = crate::fim_v1::CONTEXT_LAYOUT.into();
        p.fim_profile = Some(serving_profile());

        let bytes = container(&p);
        let trailer_len = u32::from_le_bytes(
            bytes[bytes.len() - 20..bytes.len() - 16]
                .try_into()
                .unwrap(),
        );
        assert!(trailer_len > 4096);
        assert!(trailer_len <= MAX_METADATA);
        let parsed = Payload::read(&mut Cursor::new(bytes)).unwrap();
        assert_eq!(parsed.alias, "q25-fim");
        let parsed_profile = parsed.fim_profile.unwrap();
        assert_eq!(
            parsed_profile.tokenizer.tokenizer_vocab_ids.len(),
            crate::fim_v1::FIM_SUFFIX_TOKEN_ID as usize + 1
        );

        let mut tampered = container(&p);
        tampered[4100] ^= 1;
        assert!(Payload::read(&mut Cursor::new(tampered)).is_err());
    }
    #[test]
    fn fim_profile_is_required_and_bound_to_metadata() {
        let mut p = profile();
        p.alias = "q25-fim".into();
        p.protocol = crate::fim_v1::WIRE_VERSION.into();
        p.output_tokens = crate::fim_v1::OUTPUT_TOKEN_CAP;
        p.context_layout = crate::fim_v1::CONTEXT_LAYOUT.into();
        assert!(Payload::read(&mut Cursor::new(container(&p))).is_err());

        p.fim_profile = Some(serving_profile());
        let mut invalid = container(&p);
        let metadata_len = u32::from_le_bytes(
            invalid[invalid.len() - 20..invalid.len() - 16]
                .try_into()
                .unwrap(),
        ) as usize;
        let metadata_start = invalid.len() - 20 - metadata_len;
        let mut metadata: serde_json::Value =
            serde_json::from_slice(&invalid[metadata_start..metadata_start + metadata_len])
                .unwrap();
        metadata["fim_profile"]["tokenizer"]["tokenizer_vocab_ids"][0] = serde_json::json!(1);
        let replacement = serde_json::to_vec(&metadata).unwrap();
        assert_eq!(replacement.len(), metadata_len);
        invalid[metadata_start..metadata_start + metadata_len].copy_from_slice(&replacement);
        assert!(Payload::read(&mut Cursor::new(invalid)).is_err());

        let mut unsorted = profile();
        unsorted.alias = "q25-fim".into();
        unsorted.protocol = crate::fim_v1::WIRE_VERSION.into();
        unsorted.output_tokens = crate::fim_v1::OUTPUT_TOKEN_CAP;
        unsorted.context_layout = crate::fim_v1::CONTEXT_LAYOUT.into();
        let mut invalid_profile = serving_profile();
        invalid_profile.tokenizer.special_tokens.swap(2, 3);
        unsorted.fim_profile = Some(invalid_profile);
        assert!(Payload::read(&mut Cursor::new(container(&unsorted))).is_err());
    }
    #[test]
    fn opens_the_appended_gguf_in_place_without_extracting_a_sidecar() {
        use std::{
            ffi::CString,
            fs::OpenOptions,
            io::Write,
            os::unix::ffi::OsStrExt,
            time::{SystemTime, UNIX_EPOCH},
        };

        let path = std::env::temp_dir().join(format!(
            "tabcomplete-embedded-{}-{}.bin",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let bytes = container(&profile());
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap()
            .write_all(&bytes)
            .unwrap();

        let payload = Payload::read(&mut Cursor::new(&bytes)).unwrap();
        let path_c = CString::new(path.as_os_str().as_bytes()).unwrap();
        let stream = payload.open_stream_at(&path_c).unwrap();
        assert_eq!(
            unsafe { libc::ftello(stream.0.as_ptr()) },
            payload.offset as libc::off_t
        );
        let mut header = [0; 4];
        assert_eq!(
            unsafe {
                libc::fread(
                    header.as_mut_ptr().cast(),
                    1,
                    header.len(),
                    stream.0.as_ptr(),
                )
            },
            header.len()
        );
        assert_eq!(&header, b"GGUF");
        assert!(path.exists());
        assert!(!path.with_extension("gguf").exists());
        drop(stream);
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn rejects_truncation_tampering_and_bounds() {
        let good = container(&profile());
        for length in [0, 19, 4096, good.len() - 1] {
            assert!(Payload::read(&mut Cursor::new(&good[..length])).is_err());
        }
        let mut tampered = good.clone();
        tampered[4100] ^= 1;
        assert!(Payload::read(&mut Cursor::new(tampered)).is_err());
        for mutate in [0, 1, 2, 3, 4, 5, 6] {
            let mut p = profile();
            match mutate {
                0 => p.offset += 1,
                1 => p.offset = u64::MAX,
                2 => p.length = u64::MAX,
                3 => p.output_tokens = 512,
                4 => p.alias = "unknown".into(),
                5 => p.sha256 = "0".repeat(64),
                _ => p.context_size = 8192,
            }
            assert!(Payload::read(&mut Cursor::new(container(&p))).is_err());
        }
        let mut oversized = good;
        let n = oversized.len();
        oversized[n - 20..n - 16].copy_from_slice(&(MAX_METADATA + 1).to_le_bytes());
        assert!(Payload::read(&mut Cursor::new(oversized)).is_err());
    }
}
