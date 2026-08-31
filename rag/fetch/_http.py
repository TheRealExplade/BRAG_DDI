"""Shared HTTP helper for the fetch_*.py scripts in this directory.

Both openFDA and DDInter are plain public HTTPS APIs with no auth. The one
wrinkle: some sandboxed / corporate-proxied environments cannot complete a
TLS handshake because the local machine can't reach a certificate
revocation (CRL/OCSP) endpoint, even though the underlying connection to
the real server is fine. Verified on this project's dev machine: curl and
python's requests both fail cert verification against api.fda.gov,
eutils.ncbi.nlm.nih.gov, and even google.com, while DNS resolves to the
real public IPs and the request succeeds the instant verification is
skipped -- i.e. a local trust-store gap, not a MITM.

verify=True is still the DEFAULT here. Skipping verification is opt-in only
(--insecure on each script), and prints a loud warning every time, so it
can never silently become the normal way this repo talks to the internet.
"""

import sys

import requests
import urllib3


def get(url, insecure=False, **kwargs):
    if insecure:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    try:
        return requests.get(url, timeout=30, verify=not insecure, **kwargs)
    except requests.exceptions.SSLError:
        if insecure:
            raise
        print(
            "\nTLS certificate verification failed reaching "
            f"{url.split('/')[2]}.\n"
            "If you're in a sandboxed/corporate-proxy environment (verified "
            "issue on this repo's dev machine), re-run with --insecure.\n"
            "Do NOT use --insecure on a network you don't trust.\n",
            file=sys.stderr,
        )
        raise
