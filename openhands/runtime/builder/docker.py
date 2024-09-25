import sys

import docker

from openhands.core.logger import openhands_logger as logger
from openhands.runtime.builder.base import RuntimeBuilder


class DockerRuntimeBuilder(RuntimeBuilder):
    def __init__(self, docker_client: docker.DockerClient):
        self.docker_client = docker_client

    def build(self, path: str, tags: list[str]) -> str:
        target_image_hash_name = tags[0]
        target_image_repo, target_image_hash_tag = target_image_hash_name.split(':')
        target_image_tag = tags[1].split(':')[1] if len(tags) > 1 else None

        try:
            build_logs = self.docker_client.api.build(
                path=path,
                tag=target_image_hash_name,
                rm=True,
                decode=True,
            )
        except docker.errors.BuildError as e:
            logger.error(f'Sandbox image build failed: {e}')
            raise RuntimeError(f'Sandbox image build failed: {e}')

        layers: dict[str, dict[str, str]] = {}
        previous_layer_count = 0
        for log in build_logs:
            if 'stream' in log:
                logger.debug(log['stream'].strip())
            elif 'error' in log:
                logger.error(log['error'].strip())
            elif 'status' in log:
                self._output_build_progress(log, layers, previous_layer_count)
                previous_layer_count = len(layers)
            else:
                logger.debug(str(log))

        logger.info(f'Image [{target_image_hash_name}] build finished.')

        assert (
            target_image_tag
        ), f'Expected target image tag [{target_image_tag}] is None'
        image = self.docker_client.images.get(target_image_hash_name)
        image.tag(target_image_repo, target_image_tag)
        logger.info(
            f'Re-tagged image [{target_image_hash_name}] with more generic tag [{target_image_tag}]'
        )

        # Check if the image is built successfully
        image = self.docker_client.images.get(target_image_hash_name)
        if image is None:
            raise RuntimeError(
                f'Build failed: Image {target_image_hash_name} not found'
            )

        tags_str = (
            f'{target_image_hash_tag}, {target_image_tag}'
            if target_image_tag
            else target_image_hash_tag
        )
        logger.info(
            f'Image {target_image_repo} with tags [{tags_str}] built successfully'
        )
        return target_image_hash_name

    def image_exists(self, image_name: str) -> bool:
        """Check if the image exists in the registry (try to pull it first) or in the local store.

        Args:
            image_name (str): The Docker image to check (<image repo>:<image tag>)
        Returns:
            bool: Whether the Docker image exists in the registry or in the local store
        """
        if not image_name:
            logger.error(f'Invalid image name: `{image_name}`')
            return False

        try:
            self.docker_client.images.get(image_name)
            logger.info(f'Image found locally:\n{image_name}')
            return True
        except docker.errors.ImageNotFound:
            try:
                logger.info(
                    f'Image not found locally: {image_name}.\n'
                    'Trying to pull it, please wait...'
                )

                layers: dict[str, dict[str, str]] = {}
                previous_layer_count = 0
                for line in self.docker_client.api.pull(
                    image_name, stream=True, decode=True
                ):
                    self._output_build_progress(line, layers, previous_layer_count)
                    previous_layer_count = len(layers)
                logger.debug('Image pulled')
                return True
            except docker.errors.ImageNotFound:
                logger.info('Could not find image locally or in registry.')
                return False
            except Exception as e:
                msg = 'Image could not be pulled: '
                ex_msg = str(e)
                if 'Not Found' in ex_msg:
                    msg += 'image not found in registry.'
                else:
                    msg += f'{ex_msg}'
                logger.warning(msg)
                return False

    def _output_build_progress(
        self, current_line: dict, layers: dict, previous_layer_count: int
    ) -> None:
        if 'id' in current_line and 'progressDetail' in current_line:
            layer_id = current_line['id']
            if layer_id not in layers:
                layers[layer_id] = {'status': '', 'progress': '', 'last_logged': 0}

            if 'status' in current_line:
                layers[layer_id]['status'] = current_line['status']

            if 'progress' in current_line:
                layers[layer_id]['progress'] = current_line['progress']

            if (
                'total' in current_line['progressDetail']
                and 'current' in current_line['progressDetail']
            ):
                total = current_line['progressDetail']['total']
                current = current_line['progressDetail']['current']
                percentage = (current / total) * 100
            else:
                percentage = 0

            # refresh process bar in console if stdout is a tty
            if sys.stdout.isatty():
                sys.stdout.write('\033[F' * previous_layer_count)
                for lid, layer_data in sorted(layers.items()):
                    sys.stdout.write('\033[K')
                    print(
                        f'Layer {lid}: {layer_data["progress"]} {layer_data["status"]}'
                    )
                sys.stdout.flush()
            # otherwise Log only if percentage is at least 10% higher than last logged
            elif percentage != 0 and percentage - layers[layer_id]['last_logged'] >= 10:
                logger.info(
                    f'Layer {layer_id}: {layers[layer_id]["progress"]} {layers[layer_id]["status"]}'
                )

            layers[layer_id]['last_logged'] = percentage
        elif 'status' in current_line:
            logger.info(current_line['status'])
