use std::env;
use std::io::{self, Read, Write};
use std::os::fd::FromRawFd;
use std::process::ExitCode;

const COMMAND_FD: i32 = 3;
const RESPONSE_FD: i32 = 4;
const COMMAND_BYTE_CAP: usize = 70000;
const RESPONSE_BYTE_CAP: usize = 16 * 1024 * 1024;

extern "C" {
    fn fcntl(fd: i32, command: i32, ...) -> i32;
}

const NO_BROKER: &[u8] = b"{\"format\":\"weightless.response.v1\",\"command\":{\"action\":\"\",\"key\":\"\"},\"outcome\":\"error\",\"result\":null,\"error\":{\"code\":\"broker_unavailable\",\"message\":\"Operator broker pipes are required on file descriptors 3 and 4\"},\"next\":[]}\n";

const UNAUTHORIZED: &[u8] = b"{\"format\":\"weightless.response.v1\",\"command\":{\"action\":\"\",\"key\":\"\"},\"outcome\":\"error\",\"result\":null,\"error\":{\"code\":\"unauthorized\",\"message\":\"Database, socket, docker, and verifier arguments are refused\"},\"next\":[]}\n";
const BAD_ARGV: &[u8] = b"{\"format\":\"weightless.response.v1\",\"command\":{\"action\":\"\",\"key\":\"\"},\"outcome\":\"error\",\"result\":null,\"error\":{\"code\":\"invalid_request\",\"message\":\"This program accepts no arguments\"},\"next\":[]}\n";
const BAD_SIZE: &[u8] = b"{\"format\":\"weightless.response.v1\",\"command\":{\"action\":\"\",\"key\":\"\"},\"outcome\":\"error\",\"result\":null,\"error\":{\"code\":\"invalid_request\",\"message\":\"Command exceeds byte bound\"},\"next\":[]}\n";

const PRIVILEGED: &[&str] = &[
    "--database",
    "--db",
    "--session",
    "--socket",
    "--docker",
    "--docker-host",
    "--verifier",
    "--unix-socket",
    "--verifier-db",
];

fn refuse(args: &[String]) -> Option<&'static [u8]> {
    if args.is_empty() {
        return None;
    }
    for arg in args {
        let name = arg.split_once('=').map(|(head, _)| head).unwrap_or(arg);
        if PRIVILEGED.contains(&name) {
            return Some(UNAUTHORIZED);
        }
    }
    Some(BAD_ARGV)
}

fn read_capped(mut input: impl Read, cap: usize) -> io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 8192];
    loop {
        let n = input.read(&mut tmp)?;
        if n == 0 {
            return Ok(buf);
        }
        if buf.len() + n > cap {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "cap"));
        }
        buf.extend_from_slice(&tmp[..n]);
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args == ["--help"] {
        println!("wl copies one weightless.command.v1 document from stdin and writes one weightless.response.v1 document.\n\nUsage: wl < command.json\n\nThe operator command is wsm. This binary does not open a session or choose a policy.\nAn operator must supply the broker pipes on file descriptors 3 and 4. See docs/agent.md.\n\nOptions: --help, --version");
        return ExitCode::SUCCESS;
    }
    if args == ["--version"] {
        println!("wl {}", env!("CARGO_PKG_VERSION"));
        return ExitCode::SUCCESS;
    }
    if let Some(message) = refuse(&args) {
        let _ = io::stdout().write_all(message);
        return ExitCode::from(2);
    }
    let command = match read_capped(io::stdin(), COMMAND_BYTE_CAP) {
        Ok(command) => command,
        Err(_) => {
            let _ = io::stdout().write_all(BAD_SIZE);
            return ExitCode::from(2);
        }
    };
    // F_GETFD validates ownership before constructing File from inherited descriptors.
    if unsafe { fcntl(COMMAND_FD, 1) } < 0 || unsafe { fcntl(RESPONSE_FD, 1) } < 0 {
        let _ = io::stdout().write_all(NO_BROKER);
        return ExitCode::from(1);
    }
    let mut command_pipe = unsafe { std::fs::File::from_raw_fd(COMMAND_FD) };
    if command_pipe.write_all(&command).is_err() {
        return ExitCode::from(1);
    }
    drop(command_pipe);
    let response_pipe = unsafe { std::fs::File::from_raw_fd(RESPONSE_FD) };
    let response = match read_capped(response_pipe, RESPONSE_BYTE_CAP) {
        Ok(response) => response,
        Err(_) => return ExitCode::from(1),
    };
    if io::stdout().write_all(&response).is_err() {
        return ExitCode::from(1);
    }
    ExitCode::from(0)
}
