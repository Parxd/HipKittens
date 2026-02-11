
## Setup mi300x

Load docker:
```bash
podman run -it --privileged --network=host --ipc=host \
  -v $(pwd):/workdir \
  --workdir /workdir docker.io/rocm/pytorch \
  bash
```

Or:
```bash
docker run -it --privileged --network=host --ipc=host \
  -v $(pwd):/workdir \
  --workdir /workdir docker.io/rocm/pytorch \
  bash
```




