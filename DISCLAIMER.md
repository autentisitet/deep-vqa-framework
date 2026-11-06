# Disclaimer

**Last Updated:** October 17, 2026
**Applies to:** deep-vqa-framework v0.9.6 and later

---

## 1. Data Integrity & Usage

- **Datasets**: This framework provides automated scripts to download publicly available datasets (e.g., TID2013, KoNViD-1k). The author **does not host, distribute, or store** any of these datasets and assumes no responsibility for their availability, integrity, or compliance with the original licenses provided by their respective creators.
- **User Responsibility**: Users are solely responsible for ensuring their use of these datasets complies with the Terms of Service of the hosting platforms (e.g., Google Drive, university servers) and the licensing terms specified by the dataset authors.
- **Project License Boundary**: The MIT license applies to this framework's source code only. It does not grant rights to redistribute dataset media, MOS labels, papers, or third-party model weights.
- **Checkpoint Boundary**: A checkpoint trained with a restricted or unclear dataset must not be assumed to be commercially usable. Users must review the applicable dataset and model terms before redistribution or commercial deployment.

---

## 2. Dependency Management

- **Third-Party Tools**: This framework integrates third-party downloaders such as `aria2` and `gdown`. The author is not responsible for any network security issues, data corruption, IP rate-limiting, or unexpected behavior caused by the use of these tools or external mirror sources.
- **Environment Isolation**: It is recommended to use `uv` or `venv` to isolate Python dependencies. The author is not liable for system-level conflicts, configuration errors, or hardware issues arising from installation or training on local workstations or cloud instances (e.g., AutoDL).

---

## 3. Resource Consumption

- **Cloud Billing**: Training deep learning models is resource-intensive. Users are fully responsible for monitoring their cloud GPU usage and associated costs (e.g., AutoDL credits/billing). The author is not liable for any financial charges incurred during the use of this framework.
- **Hardware Safety**: Prolonged training can generate significant heat and stress on GPU hardware. Users should ensure their systems are adequately cooled and configured for sustained high-load operations.

---

## 4. Model and service safety

- Quality scores are statistical predictions and must not be used as the sole basis for medical, legal, financial, safety, insurance, content enforcement, or other high-impact decisions.
- Authentication and rate limiting depend on the selected deployment mode. Internal mode (the default) requires an account (email + password, scrypt-hashed) and applies Nginx request limits plus a per-client failed-login limit, but does not terminate TLS and offers no email verification or self-service password recovery; an operator with shell access can reset a password from inside the container. Proxy mode disables application login; the process in front of it (which owns the reachable IP and terminates TLS) must provide authentication and rate limiting. Neither mode performs malware scanning. Add the missing controls before exposing the service to an untrusted network.

---

## 5. No Warranty

- This software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.
