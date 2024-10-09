import asyncio
from dataclasses import dataclass
import logging
from docker import DockerClient

from oh.docker.image_source.image_source_abc import ImageSourceABC
from oh.docker.util import image_exists_locally, pull_image
from oh.util.async_util import async_thread

_LOGGER = logging.getLogger(__name__)


@dataclass
class SpecificImageSource(ImageSourceABC):
    """
    Use case where we have a specific docker image we want to use (ie: Production) and we do not
    want to build one.
    """

    image_name: str

    async def get_sandbox_image(self, docker_client: DockerClient):
        return await async_thread(self.get_sandbox_image_sync, docker_client)

    def get_sandbox_image_sync(self, docker_client: DockerClient):
        if not image_exists_locally(docker_client, self.image_name):
            pull_image(docker_client, self.image_name)
        return self.image_name
