import os
import tempfile
from importlib.metadata import version
from unittest.mock import ANY, MagicMock, call, patch

import pytest
import toml
from pytest import TempPathFactory

from opendevin.runtime.utils.runtime_build import (
    RUNTIME_IMAGE_REPO,
    _generate_dockerfile,
    _get_package_version,
    _put_source_code_to_dir,
    build_runtime_image,
    get_runtime_image_repo_and_tag,
    prep_docker_build_folder,
)

OD_VERSION = f'od_v{_get_package_version()}'


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
        base_image='nikolaik/python-nodejs:python3.11-nodejs22',
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
        base_image='nikolaik/python-nodejs:python3.11-nodejs22',
        skip_init=False,
    )

    with tempfile.TemporaryDirectory() as temp_dir_2:
        dir_hash_2 = prep_docker_build_folder(
            temp_dir_2,
            base_image='nikolaik/python-nodejs:python3.11-nodejs22',
            skip_init=False,
        )
    assert dir_hash_1 == dir_hash_2


def test_hash_folder_diff_init(temp_dir):
    dir_hash_1 = prep_docker_build_folder(
        temp_dir,
        base_image='nikolaik/python-nodejs:python3.11-nodejs22',
        skip_init=False,
    )

    with tempfile.TemporaryDirectory() as temp_dir_2:
        dir_hash_2 = prep_docker_build_folder(
            temp_dir_2,
            base_image='nikolaik/python-nodejs:python3.11-nodejs22',
            skip_init=True,
        )
    assert dir_hash_1 != dir_hash_2


def test_hash_folder_diff_image(temp_dir):
    dir_hash_1 = prep_docker_build_folder(
        temp_dir,
        base_image='nikolaik/python-nodejs:python3.11-nodejs22',
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


def test_get_runtime_image_repo_and_tag_eventstream():
    base_image = 'debian:11'
    img_repo, img_tag = get_runtime_image_repo_and_tag(base_image)
    assert (
        img_repo == f'{RUNTIME_IMAGE_REPO}'
        and img_tag == f'{OD_VERSION}_image_debian_tag_11'
    )

    base_image = 'nikolaik/python-nodejs:python3.11-nodejs22'
    img_repo, img_tag = get_runtime_image_repo_and_tag(base_image)
    assert (
        img_repo == f'{RUNTIME_IMAGE_REPO}'
        and img_tag
        == f'{OD_VERSION}_image_nikolaik___python-nodejs_tag_python3.11-nodejs22'
    )

    base_image = 'ubuntu'
    img_repo, img_tag = get_runtime_image_repo_and_tag(base_image)
    assert (
        img_repo == f'{RUNTIME_IMAGE_REPO}'
        and img_tag == f'{OD_VERSION}_image_ubuntu_tag_latest'
    )


@patch('opendevin.runtime.utils.runtime_build.docker.DockerClient')
def test_build_runtime_image_from_scratch(mock_docker_client, temp_dir):
    base_image = 'debian:11'

    mock_docker_client.images.list.return_value = []

    # for image.tag(target_repo, target_image_tag)
    mock_image = MagicMock()
    mock_docker_client.images.get.return_value = mock_image

    from_scratch_hash = prep_docker_build_folder(
        temp_dir,
        base_image,
        skip_init=False,
    )

    image_name = build_runtime_image(base_image, mock_docker_client)

    # The build call should be called with the hash tag
    mock_docker_client.api.build.assert_called_once_with(
        path=ANY, tag=f'{RUNTIME_IMAGE_REPO}:{from_scratch_hash}', rm=True, decode=True
    )
    # Then the hash tag should be tagged to the version
    mock_image.tag.assert_called_once_with(
        f'{RUNTIME_IMAGE_REPO}', f'{OD_VERSION}_image_debian_tag_11'
    )
    assert image_name == f'{RUNTIME_IMAGE_REPO}:{from_scratch_hash}'


@patch('opendevin.runtime.utils.runtime_build.docker.DockerClient')
def test_build_runtime_image_exact_hash_exist(mock_docker_client, temp_dir):
    base_image = 'debian:11'

    from_scratch_hash = prep_docker_build_folder(
        temp_dir,
        base_image,
        skip_init=False,
    )

    mock_docker_client.images.list.return_value = [
        MagicMock(tags=[f'{RUNTIME_IMAGE_REPO}:{from_scratch_hash}'])
    ]

    image_name = build_runtime_image(base_image, mock_docker_client)
    assert image_name == f'{RUNTIME_IMAGE_REPO}:{from_scratch_hash}'
    mock_docker_client.api.build.assert_not_called()


@patch('opendevin.runtime.utils.runtime_build._build_sandbox_image')
@patch('opendevin.runtime.utils.runtime_build.docker.DockerClient')
def test_build_runtime_image_exact_hash_not_exist(
    mock_docker_client, mock_build_sandbox_image, temp_dir
):
    base_image = 'debian:11'
    repo, latest_image_tag = get_runtime_image_repo_and_tag(base_image)
    latest_image_name = f'{repo}:{latest_image_tag}'

    from_scratch_hash = prep_docker_build_folder(
        temp_dir,
        base_image,
        skip_init=False,
    )
    with tempfile.TemporaryDirectory() as temp_dir_2:
        non_from_scratch_hash = prep_docker_build_folder(
            temp_dir_2,
            base_image,
            skip_init=True,
        )

    # latest image exists BUT not the exact hash
    mock_docker_client.images.list.return_value = [MagicMock(tags=[latest_image_name])]

    with patch(
        'opendevin.runtime.utils.runtime_build.prep_docker_build_folder'
    ) as mock_prep_docker_build_folder:
        mock_prep_docker_build_folder.side_effect = [
            from_scratch_hash,
            non_from_scratch_hash,
        ]

        image_name = build_runtime_image(base_image, mock_docker_client)

        mock_prep_docker_build_folder.assert_has_calls(
            [
                call(ANY, base_image=base_image, skip_init=False, extra_deps=None),
                call(
                    ANY, base_image=latest_image_name, skip_init=True, extra_deps=None
                ),
            ]
        )

        mock_build_sandbox_image.assert_called_once_with(
            docker_folder=ANY,
            docker_client=mock_docker_client,
            target_image_repo=repo,
            target_image_hash_tag=from_scratch_hash,
            target_image_tag=latest_image_tag,
        )
        assert image_name == f'{repo}:{from_scratch_hash}'
