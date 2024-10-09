import logging

from docker import DockerClient

_LOGGER = logging.getLogger(__name__)


def image_exists_locally(docker_client: DockerClient, image_name: str):
    try:
        docker_client.images.get(image_name)
        _LOGGER.info(f"image_exists_locally:{image_name}")
        return True
    except:
        _LOGGER.info(f"image_does_not_exist_locally:{image_name}")
        return False


def pull_image(docker_client: DockerClient, image_name: str):
    _LOGGER.info(f"pulling_image:{image_name}")
    if ":" in image_name:
        image_repo, image_tag = image_name.split(":", 1)
    else:
        image_repo = image_name
        image_tag = None

    for line in docker_client.api.pull(
        image_repo, tag=image_tag, stream=True, decode=True
    ):
        progress_detail = line.get("progressDetail")
        if not progress_detail:
            continue
        current = progress_detail.get("current")
        total = progress_detail.get("total")
        if current and total:
            _LOGGER.debug("pull_progress:{current}/{total}")
    _LOGGER.info("image_pulled")
