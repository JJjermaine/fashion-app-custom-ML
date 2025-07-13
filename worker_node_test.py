import resource
import ray

ray.init(address='auto')

@ray.remote
def get_ulimit():
    return resource.getrlimit(resource.RLIMIT_NOFILE)

print(ray.get(get_ulimit.remote()))
