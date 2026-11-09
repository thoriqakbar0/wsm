# Local sandbox boundaries

`BubblewrapRunner` runs the existing numerical worker on Linux without Docker or KVM. Bubblewrap constructs the filesystem and namespaces. A transient systemd service owns the workload's cgroup, resource ceilings, and termination. The numerical request and response formats remain unchanged.

## Trust and authority

The trusted host owns sessions, the private verifier registry, teacher credentials, and permission to create system services. Give an untrusted agent only observations and `Session.dispatch` commands. A runner object, host shell, or system-manager connection grants privileged orchestration authority.

The threat is arbitrary Python in a probe, including attempts to read host files, reach host services, exhaust resources, fork detached descendants, or manipulate output. The host administrator, kernel, systemd, bubblewrap, installed Python runtime, and numerical worker are trusted. This runner shares the host kernel. It does not provide a microVM boundary or protection against a kernel exploit.

The local profile exposes the trusted runtime under `/usr` read-only, plus the configured Python environment when needed. Keep credentials and private datasets outside those runtime directories. An environment directory is a grant of access to its entire contents. Administrator changes to runtime libraries remain trusted changes, rather than an immutable container-image identity.

The paths and data crossing each boundary are explicit.

| Boundary | Access |
| --- | --- |
| Host to probe | Bounded stdin and Python source. No model mount, session database, verifier database, or host environment. |
| Host to numerical worker | One JSON request, a private copy of the worker code, and the configured public model directory read-only when required. |
| Verifier to final worker | Only the frozen candidate and cases for that grading operation. The registry database and other datasets stay on the host. |
| Worker to host | Bounded stdout and stderr. Invalid transport or incomplete execution cannot produce a successful result. |

The numerical runtime is trusted to process private final cases. Probe code is never inserted into a final numerical worker. A receipt contains unsigned observations and consistency evidence. It is not independent attestation of arithmetic or sandbox integrity.

## Enforced local profile

Each operation receives separate user, mount, PID, network, IPC, UTS, and cgroup namespaces. The guest uses UID and GID 65534 with capabilities dropped and no new privileges. It sees a private `/proc`, minimal devices, a read-only root and runtime, and a bounded writable `/tmp`. No host home directory, system bus, Docker socket, or network interface is passed through.

UID 65534 is a namespace identity mapped to the launcher's host UID, not a claim that the host process runs as the host's nobody account. The privileged system manager and bubblewrap setup remain trusted. This host's Ubuntu policy prevents unprivileged namespace setup. The runner does not weaken that policy.

The system manager enforces memory, swap, CPU, and process ceilings for the complete workload cgroup. A process that starts a new session remains inside that cgroup. The supervisor removes the workload on normal completion as well as timeout, output overflow, callback cancellation, and failure. Cleanup uncertainty is an error, never a successful result.

A trusted startup handshake precedes the dispatch callback and delivery of untrusted input. One deadline includes startup and execution. Teardown has a separate bounded grace period. Each input and each output stream is limited to 4 MiB. Output is read incrementally so a producer cannot postpone detection by writing a small overflow and then sleeping.

Service-manager environment expansion is disabled before any submitted source crosses the launch boundary. Clearing the guest environment alone would not prevent expansion in command arguments before guest startup.

`Session.cancel` remains a selection barrier and reports `termination_confirmed: false`. It does not synchronously interrupt an operation running through another session connection. The runner's deadline still applies. Interrupting an active runner call invokes its teardown path. These are different contracts.

## Run on this host

The local profile requires Linux namespaces, cgroup v2, bubblewrap with sized tmpfs support, and a running systemd system manager. The trusted caller needs permission to create and stop system services. The current implementation is tested with a root supervisor. It does not change host namespace sysctls or fall back to unsandboxed execution when setup fails.

Install the explicit system dependency through the distribution package manager.

```sh
apt-get install bubblewrap
```

Construct the runner without a container image.

```python
from wsm import BubblewrapRunner

runner = BubblewrapRunner(timeout=60, cpus=2, memory_bytes=2 * 1024**3)
print(runner.probe("print('isolated')"))
# Pass runner to Session.create and Verifier as usual.
```

The core Python package still has no third-party runtime dependencies. SmolLM execution additionally needs the existing pinned Torch and Transformers dependencies in the configured Python environment, plus a model staged with `stage_smollm`. No model download or installation occurs implicitly. Use `python=` to select that environment.

The existing `DockerRunner` remains available for callers and persisted sessions bound to its configuration. Local execution never falls back to it. The acceptance driver defaults to bubblewrap. Docker acceptance now requires `--runner docker --image ...` explicitly.

Run actual local isolation and binary64 grading without downloading model packages.

```sh
WEIGHTLESS_LOCAL_SANDBOX=1 PYTHONPATH=packages/python/src \
  python3 -m unittest discover -s packages/python/tests -p test_local_sandbox.py -v
PYTHONPATH=packages/python/src python3 scripts/client_native.py \
  --binary64-only --output /tmp/weightless-local-proof
```

The native tests exercise host canaries and a live host listener, namespace identity, capabilities, read-only mounts, bounded input and output, memory and process exhaustion, temporary-storage exhaustion, timeout, detached descendants, callback interruption, and private final grading with replay. Configured values alone do not establish these protections. A skipped native suite is not isolation evidence. The binary64-only driver reports SmolLM as untested.

## Upstream references and decisions

The user identified Rivet as rivet.dev and required execution on this host without Docker. The source review used pinned checkouts rather than assuming that orchestration provides isolation.

- [Rivet sandbox adapter at `21c248b`](https://github.com/rivet-dev/agents/blob/21c248b1d1d8c3a9d6c1cc7b85531a4a2d0e5c50/packages/sandbox-adapter/src/docker.ts) distinguishes terminating a control client from terminating its workload and cleans up failed container starts. The local runner likewise owns a workload independently of its client process.
- [Rivet AgentOS at `11ee7ad`](https://github.com/rivet-dev/agentos/blob/11ee7ada8bd43d59bb622f51ce1a1ab613a339f9/README.md) uses WebAssembly and V8. Native PyTorch compatibility has not been established, so it was not substituted for the numerical runtime.
- [smol-machines/smolvm security model at `8b851e6`](https://github.com/smol-machines/smolvm/blob/8b851e6f357bc4e00d3b12060ad7dfa1e638a15a/docs/security-model.md) treats every mount, forwarded socket, and network grant as workload authority. Its separate guest kernel would be a stronger boundary, but Linux execution requires KVM, which this host does not expose. The user required this host and approved bubblewrap instead.
- [Bubblewrap's security model](https://github.com/containers/bubblewrap#sandbox-security) makes the calling application's policy responsible for isolation. This implementation supplies that policy and uses systemd cgroups for the limits bubblewrap alone does not provide.

Support for other operating systems, unprivileged system-manager access, or a future VM backend remains outside this verified profile.
