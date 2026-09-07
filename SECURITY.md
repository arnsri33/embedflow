# Security policy

## Scope

EmbedFlow is an open-source research and infrastructure project. Please do not
include API keys, model tokens, credentials, private datasets, customer data,
or full production indexes in issues or pull requests.

## Reporting a vulnerability

If you find a security issue, avoid opening a public issue with an exploit or
secret. Contact the maintainers privately through the security contact listed
on the GitHub repository, or use GitHub's private vulnerability reporting when
it is enabled. Include a minimal reproduction, affected version, and impact.

Until a security contact is configured, use a private channel with the
repository owner rather than committing a credential to the repository.

## Secret-handling guidance

- Keep `.env` files and credentials outside the repository.
- Pass Qdrant and Hugging Face credentials through environment variables.
- Rotate any credential that was ever committed, even if the commit is later
  removed.
- Review generated reports and telemetry before sharing them publicly.
