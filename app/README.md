```
docker build . -t asia-northeast3-docker.pkg.dev/mlops-projects-456022/docker/mlops-app
```
- docker build .: Builds a Docker image from the Dockerfile in the current directory.
- -t asia-northeast3-docker.pkg.dev/mlops-projects-456022/docker/mlops-app: Tags the image with the specified name, pointing to your Artifact Registry repository in the asia-northeast3 region.