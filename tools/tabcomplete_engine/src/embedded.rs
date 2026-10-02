//! A seekable, mmap-compatible GGUF appended to the executable, never extracted.
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    ptr::NonNull,
};

const MAGIC: &[u8; 16] = b"TABCOMPLETEGGUF1";
const TRAILER_BYTES: u64 = 20;
const MAX_METADATA: u32 = 4096;
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
        ensure!(
            matches!(
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
            payload.context_size == 2304
                && payload.input_tokens == 1024
                && payload.microbatch_size == 64
                && payload.batch_size == 256
                && payload.threads == 4
                && payload.cache_type == "f16"
                && payload.context_layout == "cursor-last-v1",
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
        let pointer = unsafe { libc::fopen(c"/proc/self/exe".as_ptr(), c"rb".as_ptr()) };
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
        }
    }
    fn container(p: &Payload) -> Vec<u8> {
        let mut bytes = vec![0; 4096];
        bytes[..4].copy_from_slice(b"\x7fELF");
        bytes.extend(b"GGUFtest");
        let metadata = serde_json::to_vec(p).unwrap();
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
        }
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
        oversized[n - 20..n - 16].copy_from_slice(&4097u32.to_le_bytes());
        assert!(Payload::read(&mut Cursor::new(oversized)).is_err());
    }
}
