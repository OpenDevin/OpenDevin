import os
import tempfile
from importlib.metadata import version
from unittest.mock import ANY, MagicMock, patch

import pytest
import toml
from pytest import TempPathFactory

from opendevin.runtime.utils.runtime_build import (
    _generate_dockerfile,
    _get_package_version,
    _put_source_code_to_dir,
    build_runtime_image,
    get_new_image_name,
    prep_docker_build_folder,
)

OD_VERSION = f'od_v{_get_package_version()}'
RUNTIME_IMAGE_PREFIX = 'od_runtime'


@pytest.fixture(autouse=True)
def print_method_name(request):
    print('\n########################################################################')
    print(f'Running test: {request.node.name}')
    print('########################################################################')
    yield


@pytest.fixture
def temp_dir(tmp_path_factory: TempPathFactory) -> str:
    return str(tmp_path_factory.mktemp('test_runtime_build'))


def _check_source_code_in_dir(temp_dir):
    # assert there is a folder called 'code' in the temp_dir
    code_dir = os.path.join(temp_dir, 'code')
    assert os.path.exists(code_dir)
    assert os.path.isdir(code_dir)

    # check the source file is the same as the current code base
    assert os.path.exists(os.path.join(code_dir, 'pyproject.toml'))

    # The source code should only include the `opendevin` folder, but not the other folders
    assert set(os.listdir(code_dir)) == {
        'opendevin',
        'pyproject.toml',
        'poetry.lock',
        'LICENSE',
        'README.md',
        'PKG-INFO',
    }
    assert os.path.exists(os.path.join(code_dir, 'opendevin'))
    assert os.path.isdir(os.path.join(code_dir, 'opendevin'))

    # make sure the version from the pyproject.toml is the same as the current version
    with open(os.path.join(code_dir, 'pyproject.toml'), 'r') as f:
        pyproject = toml.load(f)

    _pyproject_version = pyproject['tool']['poetry']['version']
    assert _pyproject_version == version('opendevin')


def test_put_source_code_to_dir(temp_dir):
    _put_source_code_to_dir(temp_dir)
    _check_source_code_in_dir(temp_dir)


def test_docker_build_folder(temp_dir):
    prep_docker_build_folder(
        temp_dir,
        base_image='ubuntu:22.04',
        skip_init=False,
    )

    # check the source code is in the folder
    _check_source_code_in_dir(temp_dir)

    # Now check dockerfile is in the folder
    dockerfile_path = os.path.join(temp_dir, 'Dockerfile')
    assert os.path.exists(dockerfile_path)
    assert os.path.isfile(dockerfile_path)

    # check the folder only contains the source code and the Dockerfile
    assert set(os.listdir(temp_dir)) == {'code', 'Dockerfile'}


def test_hash_folder_same(temp_dir):
    dir_hash_1 = prep_docker_build_folder(
        temp_dir,
        base_image='ubuntu:22.04',
        skip_init=False,
    )

    with tempfile.TemporaryDirectory() as temp_dir_2:
        dir_hash_2 = prep_docker_build_folder(
            temp_dir_2,
            base_image='ubuntu:22.04',
            skip_init=False,
        )
    assert dir_hash_1 == dir_hash_2


def test_hash_folder_diff_init(temp_dir):
    dir_hash_1 = prep_docker_build_folder(
        temp_dir,
        base_image='ubuntu:22.04',
        skip_init=False,
    )

    with tempfile.TemporaryDirectory() as temp_dir_2:
        dir_hash_2 = prep_docker_build_folder(
            temp_dir_2,
            base_image='ubuntu:22.04',
            skip_init=True,
        )
    assert dir_hash_1 != dir_hash_2


def test_hash_folder_diff_image(temp_dir):
    dir_hash_1 = prep_docker_build_folder(
        temp_dir,
        base_image='ubuntu:22.04',
        skip_init=False,
    )

    with tempfile.TemporaryDirectory() as temp_dir_2:
        dir_hash_2 = prep_docker_build_folder(
            temp_dir_2,
            base_image='debian:11',
            skip_init=False,
        )
    assert dir_hash_1 != dir_hash_2


def test_generate_dockerfile_scratch():
    base_image = 'debian:11'
    dockerfile_content = _generate_dockerfile(
        base_image,
        skip_init=False,
    )
    assert base_image in dockerfile_content
    assert 'apt-get update' in dockerfile_content
    assert 'apt-get install -y wget sudo apt-utils' in dockerfile_content
    assert (
        'RUN /opendevin/miniforge3/bin/mamba install conda-forge::poetry python=3.11 -y'
        in dockerfile_content
    )

    # Check the update command
    assert 'COPY ./code /opendevin/code' in dockerfile_content
    assert (
        '/opendevin/miniforge3/bin/mamba run -n base poetry install'
        in dockerfile_content
    )


def test_generate_dockerfile_skip_init():
    base_image = 'debian:11'
    dockerfile_content = _generate_dockerfile(
        base_image,
        skip_init=True,
    )

    # These commands SHOULD NOT include in the dockerfile if skip_init is True
    assert 'RUN apt update && apt install -y wget sudo' not in dockerfile_content
    assert (
        'RUN /opendevin/miniforge3/bin/mamba install conda-forge::poetry python=3.11 -y'
        not in dockerfile_content
    )

    # These update commands SHOULD still in the dockerfile
    assert 'COPY ./code /opendevin/code' in dockerfile_content
    assert (
        '/opendevin/miniforge3/bin/mamba run -n base poetry install'
        in dockerfile_content
    )


def test_get_new_image_name_eventstream():
    base_image = 'debian:11'
    new_image_name = get_new_image_name(base_image)
    assert new_image_name == f'{RUNTIME_IMAGE_PREFIX}:{OD_VERSION}_image_debian_tag_11'

    base_image = 'ubuntu:22.04'
    new_image_name = get_new_image_name(base_image)
    assert (
        new_image_name == f'{RUNTIME_IMAGE_PREFIX}:{OD_VERSION}_image_ubuntu_tag_22.04'
    )

    base_image = 'ubuntu'
    new_image_name = get_new_image_name(base_image)
    assert (
        new_image_name == f'{RUNTIME_IMAGE_PREFIX}:{OD_VERSION}_image_ubuntu_tag_latest'
    )


def test_get_new_image_name_eventstream_dev_mode():
    base_image = f'{RUNTIME_IMAGE_PREFIX}:{OD_VERSION}_image_debian_tag_11'
    new_image_name = get_new_image_name(base_image, dev_mode=True)
    assert (
        new_image_name == f'{RUNTIME_IMAGE_PREFIX}_dev:{OD_VERSION}_image_debian_tag_11'
    )

    base_image = f'{RUNTIME_IMAGE_PREFIX}:{OD_VERSION}_image_ubuntu_tag_22.04'
    new_image_name = get_new_image_name(base_image, dev_mode=True)
    assert (
        new_image_name
        == f'{RUNTIME_IMAGE_PREFIX}_dev:{OD_VERSION}_image_ubuntu_tag_22.04'
    )

    base_image = f'{RUNTIME_IMAGE_PREFIX}:{OD_VERSION}_image_ubuntu_tag_latest'
    new_image_name = get_new_image_name(base_image, dev_mode=True)
    assert (
        new_image_name
        == f'{RUNTIME_IMAGE_PREFIX}_dev:{OD_VERSION}_image_ubuntu_tag_latest'
    )


def test_get_new_image_name_eventstream_dev_invalid_base_image():
    with pytest.raises(ValueError):
        base_image = 'debian:11'
        get_new_image_name(base_image, dev_mode=True)

    with pytest.raises(ValueError):
        base_image = 'ubuntu:22.04'
        get_new_image_name(base_image, dev_mode=True)

    with pytest.raises(ValueError):
        base_image = 'ubuntu:latest'
        get_new_image_name(base_image, dev_mode=True)


@patch('opendevin.runtime.utils.runtime_build.docker.DockerClient')
def test_build_runtime_image_from_scratch(mock_docker_client, temp_dir):
    base_image = 'debian:11'
    mock_docker_client.images.list.return_value = []
    # for image.tag(target_repo, target_image_tag)
    mock_image = MagicMock()
    mock_docker_client.images.get.return_value = mock_image

    dir_hash = prep_docker_build_folder(
        temp_dir,
        base_image,
        skip_init=False,
    )

    image_name = build_runtime_image(base_image, mock_docker_client)

    # The build call should be called with the hash tag
    mock_docker_client.api.build.assert_called_once_with(
        path=ANY,
        tag=f'{RUNTIME_IMAGE_PREFIX}:{dir_hash}',
        rm=True,
        decode=True,
        nocache=False,
    )
    # Then the hash tag should be tagged to the version
    mock_image.tag.assert_called_once_with(
        f'{RUNTIME_IMAGE_PREFIX}', f'{OD_VERSION}_image_debian_tag_11'
    )
    assert image_name == f'{RUNTIME_IMAGE_PREFIX}:{dir_hash}'


@patch('opendevin.runtime.utils.runtime_build.docker.DockerClient')
def test_build_runtime_image_exist_no_update_source(mock_docker_client):
    base_image = 'debian:11'
    mock_docker_client.images.list.return_value = [
        MagicMock(tags=[f'{RUNTIME_IMAGE_PREFIX}:{OD_VERSION}_image_debian_tag_11'])
    ]

    image_name = build_runtime_image(base_image, mock_docker_client)
    assert image_name == f'{RUNTIME_IMAGE_PREFIX}:{OD_VERSION}_image_debian_tag_11'

    mock_docker_client.api.build.assert_not_called()


@patch('opendevin.runtime.utils.runtime_build.docker.DockerClient')
def test_build_runtime_image_exist_with_update_source(mock_docker_client, temp_dir):
    base_image = 'debian:11'
    expected_new_image_tag = f'{OD_VERSION}_image_debian_tag_11'
    od_runtime_base_image = f'{RUNTIME_IMAGE_PREFIX}:{expected_new_image_tag}'

    mock_docker_client.images.list.return_value = [
        MagicMock(tags=[od_runtime_base_image])
    ]
    # for image.tag(target_repo, target_image_tag)
    mock_image = MagicMock()
    mock_docker_client.images.get.return_value = mock_image

    # call the function to get the dir_hash to calculate the new image name
    dir_hash = prep_docker_build_folder(
        temp_dir,
        od_runtime_base_image,
        skip_init=True,
    )

    # actual call to build the image
    image_name = build_runtime_image(
        base_image, mock_docker_client, update_source_code=True
    )

    # check the build call
    mock_docker_client.api.build.assert_called_once_with(
        path=ANY,
        tag=f'{RUNTIME_IMAGE_PREFIX}_dev:{dir_hash}',
        rm=True,
        decode=True,
        nocache=True,
    )
    # Then check the hash tag should be tagged to expected image tag
    mock_image.tag.assert_called_once_with(
        f'{RUNTIME_IMAGE_PREFIX}_dev', expected_new_image_tag
    )
    assert image_name == f'{RUNTIME_IMAGE_PREFIX}_dev:{dir_hash}'
