from abc import ABC, abstractmethod

from docker import DockerClient


class ImageSourceABC(ABC):
    """Source for docker images."""

    @abstractmethod
    async def get_sandbox_image(self, docker_client: DockerClient):
        """Get the sandbox image"""
