import asyncio
from asyncio import subprocess
from dataclasses import dataclass, field
import hashlib
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Optional
from docker import DockerClient
from jinja2 import Environment, FileSystemLoader

from oh.docker.image_source.image_source_abc import ImageSourceABC
from oh.docker.util import image_exists_locally, pull_image
from oh.util.async_util import async_thread, wait_all

_LOGGER = logging.getLogger(__name__)


@dataclass
class BuildImageSource(ImageSourceABC):
    """
    Use case where we want to build a docker image from our local files
    """

    base_image_name: str
    sandbox_image_name_format = "openhands_runtime_{hash}"
    cwd: str = field(default_factory=os.cwd)
    skip_init: bool = True
    extra_deps: Optional[str] = None
    target_image_name: Optional[str] = None
    _sandbox_image_name: Optional[str] = None

    async def get_sandbox_image(self, docker_client: DockerClient):
        return await async_thread(self.get_sandbox_image_sync, docker_client)

    def get_sandbox_image_sync(self, docker_client: DockerClient):
        if self._sandbox_image_name:
            return self._sandbox_image_name
        image_name = self.get_sandbox_image_name()
        if image_exists_locally(docker_client, image_name):
            self._sandbox_image_name = image_name
            return image_name
        self.build_image(image_name)
        if self.target_image_name:
            self._sandbox_image_name = self.target_image_name
        else:
            self._sandbox_image_name = image_name
        return self._sandbox_image_name

    def get_sandbox_image_name(self):
        return asyncio.run(self.get_sandbox_image_name_async)

    async def get_sandbox_image_name_async(self):
        # This operation is heavily io bound, so we briefly jump back into asyncio
        md5s = {}
        root = Path(os.getcwd())
        await wait_all(md5s(path, compiled_filter) for path in root.iterdir())
        hash = hashlib.md5()
        for path in sorted(md5s, key=lambda p: str(p)):
            hash.update(md5s[path])
        return self.sandbox_image_name_format.format(hash=hash.hexdigest())

    def build_image(self, image_name: str, docker_client: DockerClient):
        if not image_exists_locally(docker_client, self.base_image_name):
            pull_image(docker_client, self.base_image_name)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            self.build_project(temp_path)
            self.unzip_tarball(temp_path)
            self.create_dockerfile(temp_path)
            self.execute_docker_build(image_name, docker_client)
            self.retag_image(image_name, docker_client)

    def build_project(self, temp_path: Path):
        """
        Run a poetry build on the current project into the temp_path
        """
        result = subprocess.run(
            f"python -m build -s -o \"{temp_path}\" {self.cwd.replace(' ', r'\ ')}",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _LOGGER.info(result.stdout.decode())
        err_logs = result.stderr.decode()
        if err_logs:
            _LOGGER.error(err_logs)
        if result.returncode != 0:
            _LOGGER.error(f"Image build failed:\n{result}")
            raise RuntimeError(f"Image build failed:\n{result}")

    def unzip_tarball(self, temp_path: Path):
        """
        Unzip the first available tarball in the directory given to a directory named "code". (Also remove the tarball)
        """
        try:
            tarball_path = next(
                p for p in temp_path.iterdir() if p.name.endswith(".tar.gz")
            )
        except StopIteration:
            _LOGGER.error(
                f"Source distribution not found at {temp_path}. (Do you need to run `make build`?)"
            )
        _LOGGER.info(f"Source distribution created at {tarball_path}")
        shutil.unpack_archive(tarball_path, temp_path)
        os.remove(tarball_path)
        code_path = next(p for p in temp_path.iterdir() if not p.name.startswith("."))
        code_path.rename(Path(code_path.parent, "code"))

    def create_dockerfile(self, temp_path: Path):
        """Create a dockerfile in the temp directory given."""
        env = Environment(
            loader=FileSystemLoader(
                searchpath=Path(Path(__file__).parent, "SandboxDockerfile.j2")
            )
        )
        template = env.get_template("Dockerfile.j2")
        dockerfile_content = template.render(
            base_image=self.base_image_name,
            skip_init=self.skip_init,
            extra_deps=self.extra_deps or "",
        )
        with open(Path(temp_path, "Dockerfile"), "w") as file:
            file.write(dockerfile_content)
        return dockerfile_content

    def execute_docker_build(self, temp_path: Path, image_name: str):
        buildx_cmd = [
            "docker",
            "buildx",
            "build",
            "--progress=plain",
            f"--tag={image_name}",
            "--load",
        ]

        # Cache args / extra build args were never used in openhands, so I did not copy those over

        buildx_cmd.append(temp_path)  # must be last!

        try:
            process = subprocess.Popen(
                buildx_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            if process.stdout:
                for line in iter(process.stdout.readline, ""):
                    line = line.strip()
                    if line:
                        self._output_logs(line)

            return_code = process.wait()

            if return_code != 0:
                raise subprocess.CalledProcessError(
                    return_code,
                    process.args,
                    output=process.stdout.read() if process.stdout else None,
                    stderr=process.stderr.read() if process.stderr else None,
                )

        except subprocess.CalledProcessError as e:
            _LOGGER.error(f"Image build failed", exc_info=True)
            _LOGGER.error(f"Command output:\n{e.output}")
            raise
        except subprocess.TimeoutExpired:
            _LOGGER.error("Image build timed out", exc_info=True)
            raise
        except FileNotFoundError:
            _LOGGER.error(f"Python executable not found", exc_info=True)
            raise
        except PermissionError as e:
            _LOGGER.error("Build permission denied", exc_info=True)
            raise
        except Exception as e:
            _LOGGER.error(
                f"An unexpected error occurred during the build process", exc_info=True
            )
            raise

        _LOGGER.info(f"Image [{image_name}] build finished.")

    def retag_image(self, image_name: str, docker_client: DockerClient):
        if self.target_image_name:
            target_image_repo, target_image_tag = self.target_image_name.split(":")
            image = docker_client.images.get(image_name)
            image.tag(target_image_repo, target_image_tag)
            _LOGGER.info(
                f"Re-tagged image [{image_name}] with more generic tag [{self.target_image_name}]"
            )


def compiled_filter(path: Path) -> bool:
    name = path.name
    if (name.startswith(".") or name.endswith(".pyc") or name == "__pycache__"):
        return False
    if path.is_dir() and not Path(path, "__init__.py").exists():
        return False
    return True
