"""Copy the next command. This program does not import the kernel."""
import json
import sys


def emit(command):
    sys.stdout.write(json.dumps(command, separators=(',', ':')) + '\n')
    sys.stdout.flush()


def read_response():
    line = sys.stdin.readline()
    if not line:
        raise SystemExit('response missing')
    return json.loads(line)


def command_for(response, submitted, frozen, finalized):
    nxt = response.get('next') or []
    by_action = {item['command']['action']: item['command'] for item in nxt}
    if response.get('outcome') == 'unknown':
        return by_action['freeze'], submitted, frozen, finalized
    if response.get('outcome') == 'error':
        return by_action['observe'], submitted, frozen, finalized
    if response['command']['action'] == 'submit' and response['outcome'] == 'ok':
        submitted = True
    if response['command']['action'] == 'freeze' and response['outcome'] == 'ok':
        frozen = True
    if response['command']['action'] == 'finalize' and response['outcome'] == 'ok':
        finalized = True
    if not submitted and 'submit' in by_action:
        command = by_action['submit']
        tensor = next(iter(response['result']['parameters']))
        command['key'] = 'edit-1'
        command['payload']['edits'] = [{'tensor': tensor, 'index': 0, 'value': 1}]
        return command, submitted, frozen, finalized
    if submitted and not frozen and 'freeze' in by_action:
        return by_action['freeze'], submitted, frozen, finalized
    if frozen and not finalized and 'finalize' in by_action:
        return by_action['finalize'], submitted, frozen, finalized
    if finalized and 'export' in by_action:
        return by_action['export'], submitted, frozen, finalized
    return by_action['observe'], submitted, frozen, finalized


def main():
    submitted = frozen = finalized = False
    emit({'format': 'weightless.command.v1', 'action': 'observe', 'key': 'wl:observe', 'payload': {}})
    while True:
        response = read_response()
        command, submitted, frozen, finalized = command_for(response, submitted, frozen, finalized)
        emit(command)
        if command['action'] == 'export':
            return


if __name__ == '__main__':
    main()
