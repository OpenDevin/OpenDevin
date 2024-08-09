import argparse
import os
import shutil
import subprocess
import tempfile

import docker
import toml
from dirhash import dirhash
from jinja2 import Environment, FileSystemLoader

import opendevin
from opendevin.core.logger import opendevin_logger as logger

RUNTIME_IMAGE_REPO = os.getenv(
    'OD_RUNTIME_RUNTIME_IMAGE_REPO', 'ghcr.io/opendevin/od_runtime'
)


def _get_package_version():
    """Read the version from pyproject.toml as the other one may be outdated."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(opendevin.__file__)))
    pyproject_path = os.path.join(project_root, 'pyproject.toml')
    with open(pyproject_path, 'r') as f:
        pyproject_data = toml.load(f)
    return pyproject_data['tool']['poetry']['version']


def _create_project_source_dist():
    """Create a source distribution of the project. Return the path to the tarball."""
    # Copy the project directory to the container
    # get the location of "opendevin" package
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(opendevin.__file__)))
    logger.info(f'Using project root: {project_root}')

    # run "python -m build -s" on project_root
    result = subprocess.run(['python', '-m', 'build', '-s', project_root])
    if result.returncode != 0:
        logger.error(f'Build failed: {result}')
        raise Exception(f'Build failed: {result}')

    # Fetch the correct version from pyproject.toml
    package_version = _get_package_version()
    tarball_path = os.path.join(
        project_root, 'dist', f'opendevin-{package_version}.tar.gz'
    )
    if not os.path.exists(tarball_path):
        logger.error(f'Source distribution not found at {tarball_path}')
        raise Exception(f'Source distribution not found at {tarball_path}')
    logger.info(f'Source distribution created at {tarball_path}')

    return tarball_path


def _put_source_code_to_dir(temp_dir: str):
    """Put the source code of OpenDevin to the temp_dir/code."""
    tarball_path = _create_project_source_dist()
    filename = os.path.basename(tarball_path)
    filename = filename.removesuffix('.tar.gz')

    # move the tarball to temp_dir
    _res = shutil.copy(tarball_path, os.path.join(temp_dir, 'project.tar.gz'))
    if _res:
        os.remove(tarball_path)
    logger.info(
        f'Source distribution moved to {os.path.join(temp_dir, "project.tar.gz")}'
    )

    # unzip the tarball
    shutil.unpack_archive(os.path.join(temp_dir, 'project.tar.gz'), temp_dir)
    # remove the tarball
    os.remove(os.path.join(temp_dir, 'project.tar.gz'))
    # rename the directory to the 'code'
    os.rename(os.path.join(temp_dir, filename), os.path.join(temp_dir, 'code'))
    logger.info(f'Unpacked source code directory: {os.path.join(temp_dir, "code")}')


def _generate_dockerfile(
    base_image: str,
    skip_init: bool = False,
    extra_deps: str | None = None,
) -> str:
    """Generate the Dockerfile content for the eventstream runtime image based on user-provided base image."""
    env = Environment(
        loader=FileSystemLoader(
            searchpath=os.path.join(os.path.dirname(__file__), 'runtime_templates')
        )
    )
    template = env.get_template('Dockerfile.j2')
    dockerfile_content = template.render(
        base_image=base_image,
        skip_init=skip_init,
        extra_deps=extra_deps if extra_deps is not None else '',
    )
    return dockerfile_content


def prep_docker_build_folder(
    dir_path: str,
    base_image: str,
    skip_init: bool = False,
    extra_deps: str | None = None,
) -> str:
    """Prepares the docker build folder by copying the source code and generating the Dockerfile.

    Return the MD5 hash of the directory.
    """
    _put_source_code_to_dir(dir_path)
    dockerfile_content = _generate_dockerfile(
        base_image,
        skip_init=skip_init,
        extra_deps=extra_deps,
    )
    logger.info(
        (
            f'===== Dockerfile content =====\n'
            f'{dockerfile_content}\n'
            f'==============================='
        )
    )
    with open(os.path.join(dir_path, 'Dockerfile'), 'w') as file:
        file.write(dockerfile_content)

    hash = dirhash(dir_path, 'md5')
    logger.info(
        f'Input base image: {base_image}\n'
        f'Skip init: {skip_init}\n'
        f'Extra deps: {extra_deps}\n'
        f'Hash for docker build directory [{dir_path}] (contents: {os.listdir(dir_path)}): {hash}\n'
    )
    return hash


def _build_sandbox_image(
    docker_folder: str,
    docker_client: docker.DockerClient,
    target_image_repo: str,
    target_image_hash_tag: str,
    target_image_tag: str,
) -> str:
    """Build the sandbox image.

    The image will be tagged as both:
        - target_image_repo:target_image_hash_tag
        - target_image_repo:target_image_tag

    Args:
        docker_folder: str: the path to the docker build folder
        docker_client: docker.DockerClient: the docker client
        target_image_repo: str: the repository name for the target image
        target_image_hash_tag: str: the *hash* tag for the target image that is calculated based
            on the contents of the docker build folder (source code and Dockerfile)
            e.g., ubuntu:latest -> od_runtime:1234567890abcdef
        target_image_tag: str: the tag for the target image that's generic and based on the base image name
            e.g., ubuntu:latest -> od_runtime:ubuntu_tag_latest
    """
    # 1. Always directly build and tag using the dir_hash
    target_image_hash_name = f'{target_image_repo}:{target_image_hash_tag}'
    try:
        build_logs = docker_client.api.build(
            path=docker_folder,
            tag=target_image_hash_name,
            rm=True,
            decode=True,
        )
    except docker.errors.BuildError as e:
        logger.error(f'Sandbox image build failed: {e}')
        raise e

    for log in build_logs:
        if 'stream' in log:
            print(log['stream'].strip())
        elif 'error' in log:
            logger.error(log['error'].strip())
        else:
            logger.info(str(log))

    # 2. Re-tag the image with a more generic tag (as somewhat of "latest" tag)
    logger.info(f'Image [{target_image_hash_name}] build finished.')
    image = docker_client.images.get(target_image_hash_name)
    image.tag(target_image_repo, target_image_tag)
    logger.info(
        f'Re-tagged image [{target_image_hash_name}] with more generic tag [{target_image_tag}]'
    )

    # check if the image is built successfully
    image = docker_client.images.get(target_image_hash_name)
    if image is None:
        raise RuntimeError(
            f'Build failed: Image [{target_image_repo}:{target_image_hash_tag}] not found'
        )
    logger.info(
        f'Image [{target_image_repo}:{target_image_hash_tag}] (hash: [{target_image_tag}]) built successfully'
    )
    return target_image_hash_name


def get_runtime_image_repo_and_tag(base_image: str) -> tuple[str, str]:
    if RUNTIME_IMAGE_REPO in base_image:
        logger.info(
            f'The provided image [{base_image}] is a already a valid od_runtime image.\n'
            f'Will try to reuse it as is.'
        )
        if ':' not in base_image:
            base_image = base_image + ':latest'
        repo, tag = base_image.split(':')
        return repo, tag
    else:
        if ':' not in base_image:
            base_image = base_image + ':latest'
        [repo, tag] = base_image.split(':')
        repo = repo.replace('/', '___')
        od_version = _get_package_version()
        return RUNTIME_IMAGE_REPO, f'od_v{od_version}_image_{repo}_tag_{tag}'


def _check_image_exists(image_name: str, docker_client: docker.DockerClient) -> bool:
    """Check if the image exists in the registry (try to pull it first) AND in the local store.

    image_name is f'{repo}:{tag}'
    """
    # Try to pull the new image from the registry
    try:
        docker_client.images.pull(image_name)
    except Exception:
        logger.info(f'Cannot pull image {image_name} directly')

    images = docker_client.images.list()
    if images:
        for image in images:
            if image_name in image.tags:
                return True
    return False


def build_runtime_image(
    base_image: str,
    docker_client: docker.DockerClient,
    extra_deps: str | None = None,
    docker_build_folder: str | None = None,
    dry_run: bool = False,
    force_rebuild: bool = False,
) -> str:
    """Build the runtime image for the OpenDevin runtime.

    See https://docs.all-hands.dev/modules/usage/runtime for more details.
    """
    runtime_image_repo, runtime_image_tag = get_runtime_image_repo_and_tag(base_image)

    # Calculate the hash for the docker build folder (source code and Dockerfile)
    with tempfile.TemporaryDirectory() as temp_dir:
        from_scratch_hash = prep_docker_build_folder(
            temp_dir,
            base_image=base_image,
            skip_init=False,
            extra_deps=extra_deps,
        )

    # hash image name, if the hash matches, it means the image is already
    # built from scratch with the *exact SAME source code* on the exact Dockerfile
    hash_runtime_image_name = f'{runtime_image_repo}:{from_scratch_hash}'

    # non-hash generic image name, it could contains *similar* dependencies
    # but *might* not exactly match the state of the source code.
    # It resembles the "latest" tag in the docker image naming convention for
    # a particular {repo}:{tag} pair (e.g., ubuntu:latest -> od_runtime:ubuntu_tag_latest)
    # we will build from IT to save time if the `from_scratch_hash` is not found
    generic_runtime_image_name = f'{runtime_image_repo}:{runtime_image_tag}'

    # 1. If the image exists with the same hash, we will reuse it as is
    if _check_image_exists(hash_runtime_image_name, docker_client):
        logger.info(
            f'Image [{hash_runtime_image_name}] exists with matched hash for Docker build folder.\n'
            'Will reuse it as is.'
        )
        return hash_runtime_image_name

    # 2. If the exact hash is not found, we will FIRST try to re-build it
    # by leveraging the non-hash `generic_runtime_image_name` to save some time
    # from re-building the dependencies (e.g., poetry install, apt install)
    elif (
        _check_image_exists(generic_runtime_image_name, docker_client)
        and not force_rebuild
    ):
        logger.info(
            f'Cannot find matched hash for image [{hash_runtime_image_name}]\n'
            f'Will try to re-build it from latest [{generic_runtime_image_name}] image to potentially save '
            f'time for dependencies installation.\n'
        )

        cur_docker_build_folder = docker_build_folder or tempfile.mkdtemp()
        _skip_init_hash = prep_docker_build_folder(
            cur_docker_build_folder,
            # we want to use the existing generic image as base
            # so that we can leverage existing dependencies already installed in the image
            base_image=generic_runtime_image_name,
            skip_init=True,  # skip init since we are re-using the existing image
            extra_deps=extra_deps,
        )
        assert (
            _skip_init_hash != from_scratch_hash
        ), f'The skip_init hash [{_skip_init_hash}] should not match the existing hash [{from_scratch_hash}]'

        if not dry_run:
            _build_sandbox_image(
                docker_folder=cur_docker_build_folder,
                docker_client=docker_client,
                target_image_repo=runtime_image_repo,
                # NOTE: WE ALWAYS use the "from_scratch_hash" tag for the target image
                # otherwise, even if the source code is exactly the same, the image *might* be re-built
                # because the same source code will generate different hash when skip_init=True/False
                # since the Dockerfile is slightly different
                target_image_hash_tag=from_scratch_hash,
                target_image_tag=runtime_image_tag,
            )
        else:
            logger.info(
                f'Dry run: Skipping image build for [{generic_runtime_image_name}]'
            )
        if docker_build_folder is None:
            shutil.rmtree(cur_docker_build_folder)

    # 3. If the image is not found AND we cannot re-use the non-hash latest relavant image,
    # we will build it completely from scratch
    else:
        if force_rebuild:
            logger.info(
                f'Force re-build: Will try to re-build image [{generic_runtime_image_name}] from scratch.\n'
            )
        cur_docker_build_folder = docker_build_folder or tempfile.mkdtemp()
        _new_from_scratch_hash = prep_docker_build_folder(
            cur_docker_build_folder,
            base_image,
            skip_init=False,
            extra_deps=extra_deps,
        )
        assert (
            _new_from_scratch_hash == from_scratch_hash
        ), f'The new from scratch hash [{_new_from_scratch_hash}] does not match the existing hash [{from_scratch_hash}]'

        if not dry_run:
            _build_sandbox_image(
                docker_folder=cur_docker_build_folder,
                docker_client=docker_client,
                target_image_repo=runtime_image_repo,
                # NOTE: WE ALWAYS use the "from_scratch_hash" tag for the target image
                target_image_hash_tag=from_scratch_hash,
                target_image_tag=runtime_image_tag,
            )
        else:
            logger.info(
                f'Dry run: Skipping image build for [{generic_runtime_image_name}]'
            )

        if docker_build_folder is None:
            shutil.rmtree(cur_docker_build_folder)

    return f'{runtime_image_repo}:{from_scratch_hash}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_image', type=str, default='nikolaik/python-nodejs:python3.11-nodejs22'
    )
    parser.add_argument('--build_folder', type=str, default=None)
    parser.add_argument('--force_rebuild', action='store_true', default=False)
    args = parser.parse_args()

    if args.build_folder is not None:
        build_folder = args.build_folder
        assert os.path.exists(
            build_folder
        ), f'Build folder {build_folder} does not exist'
        logger.info(
            f'Will prepare a build folder by copying the source code and generating the Dockerfile: {build_folder}'
        )
        runtime_image_repo, runtime_image_tag = get_runtime_image_repo_and_tag(
            args.base_image
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime_image_hash_name = build_runtime_image(
                args.base_image,
                docker_client=docker.from_env(),
                docker_build_folder=temp_dir,
                dry_run=True,
                force_rebuild=args.force_rebuild,
            )
            _runtime_image_repo, runtime_image_hash_tag = runtime_image_hash_name.split(
                ':'
            )
            # Move contents of temp_dir to build_folder
            shutil.copytree(temp_dir, build_folder, dirs_exist_ok=True)
        logger.info(
            f'Build folder [{build_folder}] is ready: {os.listdir(build_folder)}'
        )

        with open(os.path.join(build_folder, 'config.sh'), 'a') as file:
            file.write(
                (
                    f'\n'
                    f'DOCKER_IMAGE={runtime_image_repo}\n'
                    f'DOCKER_IMAGE_TAG={runtime_image_tag}\n'
                    f'DOCKER_IMAGE_HASH_TAG={runtime_image_hash_tag}\n'
                )
            )
        logger.info(
            f'`config.sh` is updated with the new image name [{runtime_image_repo}] and tag [{runtime_image_tag}, {runtime_image_hash_tag}]'
        )
        logger.info(f'Dockerfile and source distribution are ready in {build_folder}')
    else:
        logger.info('Building image in a temporary folder')
        client = docker.from_env()
        image_name = build_runtime_image(args.base_image, client)
        print(f'\nBUILT Image: {image_name}\n')
