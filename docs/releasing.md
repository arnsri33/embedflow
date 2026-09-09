# Releasing EmbedFlow

This page covers the release sequence for maintainers. It is intentionally
separate from the user installation guide. The repository workflow never runs
for ordinary pushes or pull requests; it publishes only after a GitHub Release
is published and the `pypi` environment is approved.

## Local preflight

Run these commands from a clean checkout:

```bash
rm -rf dist build embedflow.egg-info
python -m build
python -m twine check dist/*
```

Inspect both archives before uploading:

```bash
unzip -l dist/embedflow-0.2.0-py3-none-any.whl
tar -tzf dist/embedflow-0.2.0.tar.gz
sha256sum dist/*
```

The wheel should be `py3-none-any`. It contains the `embedflow` package, the
frozen T2-v1 contract, and the registry data. Model weights, indexes, caches,
logs, credentials, and local databases stay outside the distributions.

Test the wheel outside the source tree:

```bash
python -m venv /tmp/embedflow-wheel-test
/tmp/embedflow-wheel-test/bin/python -m pip install --upgrade pip
/tmp/embedflow-wheel-test/bin/python -m pip install dist/embedflow-0.2.0-py3-none-any.whl
cd /tmp
/tmp/embedflow-wheel-test/bin/python -c "import embedflow; print(embedflow.__version__)"
/tmp/embedflow-wheel-test/bin/embedflow --help
/tmp/embedflow-wheel-test/bin/embedflow registry list
/tmp/embedflow-wheel-test/bin/embedflow doctor
```

## TestPyPI

TestPyPI is a separate service and requires a separate account. Create one at
<https://test.pypi.org/account/register/> if you do not already have it. The
same email address can be used, but the account and 2FA configuration are
separate from PyPI.

For a manual upload, install Twine in a release-only environment and use a
short-lived token or an interactive credential prompt. Never put a token in
the repository or a committed shell script:

```bash
python -m pip install build twine
python -m twine upload --repository testpypi dist/*
```

After the upload, install from TestPyPI while resolving dependencies from
production PyPI:

```bash
python -m venv /tmp/embedflow-testpypi
/tmp/embedflow-testpypi/bin/python -m pip install --upgrade pip
/tmp/embedflow-testpypi/bin/python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  embedflow==0.2.0
cd /tmp
/tmp/embedflow-testpypi/bin/python -c "import embedflow; print(embedflow.__version__)"
/tmp/embedflow-testpypi/bin/embedflow --help
/tmp/embedflow-testpypi/bin/embedflow registry list
/tmp/embedflow-testpypi/bin/embedflow doctor
```

Test optional integrations in a second clean environment:

```bash
/tmp/embedflow-testpypi/bin/python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  "embedflow[faiss,dashboard]==0.2.0"
```

If the same filename already exists on TestPyPI, use a pre-release such as
`0.2.0rc1` for the TestPyPI-only trial. Keep production `0.2.0` unchanged.

## Trusted Publishing configuration

The production workflow is `.github/workflows/release.yml`. In PyPI, open
**Account settings → Publishing → Add a new pending publisher** and enter:

| Field | Value |
| --- | --- |
| PyPI project name | `embedflow` |
| Owner | `arnsri33` |
| Repository name | `embedflow` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

Create a GitHub environment named `pypi` under **Settings → Environments** and
add a required reviewer before the first production release. The publish job
is the only job with `id-token: write`; no PyPI API token is stored in GitHub.

For TestPyPI Trusted Publishing, configure a separate pending publisher at
<https://test.pypi.org/manage/account/publishing/> with the same repository and
workflow, but use the GitHub environment name `testpypi`. The current release
workflow does not publish to TestPyPI automatically; the manual Twine path
above keeps the test step explicit.

## Production sequence

1. Build and inspect the distributions locally.
2. Upload to TestPyPI and complete the clean TestPyPI install checks.
3. Run the final release gate and review the generated report.
4. Configure the PyPI pending publisher and protected `pypi` environment.
5. Create a Git tag and GitHub Release for the exact package version, for
   example `v0.2.0`.
6. Approve the `pypi` environment when the release workflow is ready.
7. Verify the files and metadata on PyPI.
8. Install from production PyPI in a directory outside this checkout.

The GitHub workflow builds once, validates the metadata and wheel, transfers
those exact files as an artifact, and then publishes them. PyPI filenames are
immutable, so a correction after publishing requires a new version such as
`0.2.1`.

## After production publication

Once `https://pypi.org/project/embedflow/` is live, update the README install
block to:

```bash
python -m pip install "embedflow[faiss,dashboard]"
```

Keep the source-checkout instructions in the contributor documentation. The
README should not advertise the PyPI command before the first production
upload has completed.
