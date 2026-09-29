"""Choose one Kubernetes GPU resource pool without reserving other GPU types."""

import re
from collections import defaultdict


def quantity(value):
    match = re.fullmatch(r'([0-9.]+)([A-Za-z]*)', str(value))
    if not match:
        raise ValueError('Unsupported resource quantity: ' + str(value))
    number, suffix = match.groups()
    return float(number) * {'': 1, 'm': .001, 'Ki': 2**10, 'Mi': 2**20, 'Gi': 2**30,
                           'Ti': 2**40, 'K': 1e3, 'M': 1e6, 'G': 1e9, 'T': 1e12}[suffix]


def pools(nodes, pods, quotas, cfg, mode):
    used = defaultdict(lambda: defaultdict(float))
    pending = defaultdict(float)
    for pod in pods:
        if pod.get('status', {}).get('phase') in ('Succeeded', 'Failed'):
            continue
        spec = pod['spec']
        requests = defaultdict(float)
        for container in spec.get('containers', []):
            for key, value in container.get('resources', {}).get('requests', {}).items():
                requests[key] += quantity(value)
        for container in spec.get('initContainers', []):
            for key, value in container.get('resources', {}).get('requests', {}).items():
                requests[key] = max(requests[key], quantity(value))
        for key, value in requests.items():
            used[spec.get('nodeName', '')][key] += value
            if not spec.get('nodeName'):
                pending[key] += value
    quota_left = {}
    for quota in quotas:
        status = quota.get('status', {})
        for key, limit in status.get('hard', {}).items():
            if key.startswith(('requests.nvidia.com/', 'limits.nvidia.com/')):
                resource = key.split('.', 1)[1]
                left = quantity(limit) - quantity(status.get('used', {}).get(key, 0))
                quota_left[resource] = min(quota_left.get(resource, float('inf')), left)
    memory = cfg['gpu_min_memory_mib'][mode]
    result = {}
    for node in nodes:
        name, labels = node['metadata']['name'], node['metadata'].get('labels', {})
        if name in cfg['excluded_nodes'] or node['spec'].get('unschedulable'):
            continue
        if not any(c['type'] == 'Ready' and c['status'] == 'True' for c in node['status']['conditions']):
            continue
        if labels.get('kubernetes.io/arch') != 'amd64' or labels.get('kubernetes.io/os') != 'linux':
            continue
        if int(labels.get('nvidia.com/gpu.compute.major', 0)) < 8 or int(labels.get('nvidia.com/gpu.memory', 0)) < memory:
            continue
        if any(t['effect'] in ('NoSchedule', 'NoExecute') and t['key'] != 'nvidia.com/gpu'
               for t in node['spec'].get('taints', [])):
            continue
        alloc = node['status']['allocatable']
        for resource, count in alloc.items():
            if not resource.startswith('nvidia.com/') or resource == 'nvidia.com/pgpu' or 'mig' in resource or not int(count):
                continue
            if quota_left.get(resource, float('inf')) < 1:
                continue
            pool = result.setdefault(resource, {'resource': resource, 'products': set(), 'capacity': 0, 'free': 0})
            pool['products'].add(labels['nvidia.com/gpu.product'])
            pool['capacity'] += int(count)
            free = min(int(count) - used[name][resource],
                       int((quantity(alloc['cpu']) - used[name]['cpu']) / quantity(cfg['cpu'])),
                       int((quantity(alloc['memory']) - used[name]['memory']) / (cfg['memory_gi'] * 2**30)))
            pool['free'] += max(0, free)
    for pool in result.values():
        pool['products'] = sorted(pool['products'])
        pool['backlog'] = pending[pool['resource']]
        pool['free'] = min(pool['free'], quota_left.get(pool['resource'], float('inf')))
    return list(result.values())
