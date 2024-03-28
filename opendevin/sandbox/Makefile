DOCKER_BUILD_REGISTRY=ghcr.io
DOCKER_BUILD_ORG=opendevin
DOCKER_BUILD_REPO=sandbox
DOCKER_BUILD_TAG=v0.1
FULL_IMAGE=$(DOCKER_BUILD_REGISTRY)/$(DOCKER_BUILD_ORG)/$(DOCKER_BUILD_REPO):$(DOCKER_BUILD_TAG)
LATEST_FULL_IMAGE=$(DOCKER_BUILD_REGISTRY)/$(DOCKER_BUILD_ORG)/$(DOCKER_BUILD_REPO):latest

# normally, for local build testing or development. use cross platform build for sharing images to others.
build:
	docker build -f Dockerfile -t ${FULL_IMAGE} -t ${LATEST_FULL_IMAGE} .

push:
	docker push ${FULL_IMAGE} ${LATEST_FULL_IMAGE}

test:
	docker buildx build --platform linux/amd64 \
	-t ${FULL_IMAGE} -t ${LATEST_FULL_IMAGE} --load -f Dockerfile .

# cross platform build, you may need to manually stop the buildx(buildkit) container
all:
	docker buildx build --platform linux/amd64,linux/arm64 \
	-t ${FULL_IMAGE} -t ${LATEST_FULL_IMAGE} --push -f Dockerfile .

get-full-image:
	@echo ${FULL_IMAGE}
