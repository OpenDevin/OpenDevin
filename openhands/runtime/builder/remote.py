import base64
import io
import tarfile
import time

import requests

from openhands.core.logger import openhands_logger as logger
from openhands.runtime.builder import RuntimeBuilder
from openhands.runtime.utils.request import send_request_with_retry
from openhands.runtime.utils.shutdown_listener import (
    should_continue,
    sleep_if_should_continue,
)


class RemoteRuntimeBuilder(RuntimeBuilder):
    """This class interacts with the remote Runtime API for building and managing container images."""

    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({'X-API-Key': self.api_key})

    def build(self, path: str, tags: list[str]) -> str:
        """Builds a Docker image using the Runtime API's /build endpoint."""
        # Create a tar archive of the build context
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
            tar.add(path, arcname='.')
        tar_buffer.seek(0)

        # Encode the tar file as base64
        base64_encoded_tar = base64.b64encode(tar_buffer.getvalue()).decode('utf-8')

        # Prepare the multipart form data
        files = [
            ('context', ('context.tar.gz', base64_encoded_tar)),
            ('target_image', (None, tags[0])),
        ]

        # Add additional tags if present
        for tag in tags[1:]:
            files.append(('tags', (None, tag)))

        # Send the POST request to /build (Begins the build process)
        response = send_request_with_retry(
            self.session,
            'POST',
            f'{self.api_url}/build',
            files=files,
            timeout=30,
        )

        if response.status_code != 202:
            logger.error(f'Build initiation failed: {response.text}')
            raise RuntimeError(f'Build initiation failed: {response.text}')

        build_data = response.json()
        build_id = build_data['build_id']
        logger.info(f'Build initiated with ID: {build_id}')

        # Poll /build_status until the build is complete
        start_time = time.time()
        timeout = 30 * 60  # 20 minutes in seconds
        while should_continue():
            if time.time() - start_time > timeout:
                logger.error('Build timed out after 30 minutes')
                raise RuntimeError('Build timed out after 30 minutes')

            status_response = send_request_with_retry(
                self.session,
                'GET',
                f'{self.api_url}/build_status',
                params={'build_id': build_id},
                timeout=30,
            )

            if status_response.status_code != 200:
                logger.error(f'Failed to get build status: {status_response.text}')
                raise RuntimeError(
                    f'Failed to get build status: {status_response.text}'
                )

            status_data = status_response.json()
            status = status_data['status']
            logger.info(f'Build status: {status}')

            if status == 'SUCCESS':
                logger.info(f"Successfully built {status_data['image']}")
                return status_data['image']
            elif status in [
                'FAILURE',
                'INTERNAL_ERROR',
                'TIMEOUT',
                'CANCELLED',
                'EXPIRED',
            ]:
                error_message = status_data.get(
                    'error', f'Build failed with status: {status}. Build ID: {build_id}'
                )
                logger.error(error_message)
                raise RuntimeError(error_message)

            # Wait before polling again
            sleep_if_should_continue(30)

        raise RuntimeError('Build interrupted (likely received SIGTERM or SIGINT).')

    def image_exists(self, image_name: str, pull_from_repo: bool = True) -> bool:
        """Checks if an image exists in the remote registry using the /image_exists endpoint."""
        params = {'image': image_name}
        response = send_request_with_retry(
            self.session,
            'GET',
            f'{self.api_url}/image_exists',
            params=params,
            timeout=30,
        )

        if response.status_code != 200:
            logger.error(f'Failed to check image existence: {response.text}')
            raise RuntimeError(f'Failed to check image existence: {response.text}')

        result = response.json()

        if result['exists']:
            logger.info(
                f"Image {image_name} exists. "
                f"Uploaded at: {result['image']['upload_time']}, "
                f"Size: {result['image']['image_size_bytes'] / 1024 / 1024:.2f} MB"
            )
        else:
            logger.info(f'Image {image_name} does not exist.')

        return result['exists']
