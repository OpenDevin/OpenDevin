#!/bin/bash
set -eo pipefail

image_name=$1
org_name=$2
platform=$3

echo "Building: $image_name for platform: $platform"
tags=()

OPEN_DEVIN_BUILD_VERSION="dev"

if [[ -n $GITHUB_REF_NAME ]]; then
  # check if ref name is a version number
  if [[ $GITHUB_REF_NAME =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    major_version=$(echo "$GITHUB_REF_NAME" | cut -d. -f1)
    minor_version=$(echo "$GITHUB_REF_NAME" | cut -d. -f1,2)
    tags+=("$major_version" "$minor_version")
    tags+=("latest")
  fi
  sanitized=$(echo "$GITHUB_REF_NAME" | sed 's/[^a-zA-Z0-9.-]\+/-/g')
  OPEN_DEVIN_BUILD_VERSION=$sanitized
  tag=$(echo "$sanitized" | tr '[:upper:]' '[:lower:]') # lower case is required in tagging
  tags+=("$tag")
fi
echo "Tags: ${tags[@]}"

if [[ "$image_name" == "opendevin" ]]; then
  dir="./containers/app"
elif [[ "$image_name" == "od_runtime" ]]; then
  dir="./containers/runtime"
else
  dir="./containers/$image_name"
fi

if [[ (! -f "$dir/Dockerfile") && "$image_name" != "od_runtime" ]]; then
  # Allow runtime to be built without a Dockerfile
  echo "No Dockerfile found"
  exit 1
fi
if [[ ! -f "$dir/config.sh" ]]; then
  echo "No config.sh found for Dockerfile"
  exit 1
fi

source "$dir/config.sh"

if [[ -n "$org_name" ]]; then
  DOCKER_ORG="$org_name"
fi

# If $DOCKER_IMAGE_TAG is set, add it to the tags
if [[ -n "$DOCKER_IMAGE_TAG" ]]; then
  tags+=("$DOCKER_IMAGE_TAG")
fi
# If $DOCKER_IMAGE_HASH_TAG is set, add it to the tags
if [[ -n "$DOCKER_IMAGE_HASH_TAG" ]]; then
  tags+=("$DOCKER_IMAGE_HASH_TAG")
fi


DOCKER_REPOSITORY="$DOCKER_REGISTRY/$DOCKER_ORG/$DOCKER_IMAGE"
DOCKER_REPOSITORY=${DOCKER_REPOSITORY,,} # lowercase
echo "Repo: $DOCKER_REPOSITORY"
echo "Base dir: $DOCKER_BASE_DIR"

args=""
for tag in "${tags[@]}"; do
  args+=" -t $DOCKER_REPOSITORY:$tag"
done

output_image="/tmp/${image_name}-${tags[0]}-docker-image-${platform}.tar"

docker buildx build \
  $args \
  --build-arg OPEN_DEVIN_BUILD_VERSION="$OPEN_DEVIN_BUILD_VERSION" \
  --platform linux/$platform \
  --provenance=false \
  -f "$dir/Dockerfile" \
  --output type=docker,dest="$output_image" \
  "$DOCKER_BASE_DIR"

echo "${tags[*]}" > tags.txt
