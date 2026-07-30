# Disclaimer

## 1. Data Integrity & Usage

- **Datasets**: This framework provides automated scripts to download publicly available datasets (e.g., TID2013, KoNViD-1k). The author **does not host, distribute, or store** any of these datasets and assumes no responsibility for their availability, integrity, or compliance with the original licenses provided by their respective creators.
- **User Responsibility**: Users are solely responsible for ensuring their use of these datasets complies with the Terms of Service of the hosting platforms (e.g., Google Drive, university servers) and the licensing terms specified by the dataset authors.

## 2. Dependency Management

- **Third-Party Tools**: This framework integrates third-party downloaders such as `aria2` and `gdown`. The author is not responsible for any network security issues, data corruption, IP rate-limiting, or unexpected behavior caused by the use of these tools or external mirror sources.
- **Environment Isolation**: It is recommended to use `uv` or `venv` to isolate Python dependencies. The author is not liable for system-level conflicts, configuration errors, or hardware issues arising from installation or training on local workstations or cloud instances (e.g., AutoDL).

## 3. Resource Consumption

- **Cloud Billing**: Training deep learning models is resource-intensive. Users are fully responsible for monitoring their cloud GPU usage and associated costs (e.g., AutoDL credits/billing). The author is not liable for any financial charges incurred during the use of this framework.
- **Hardware Safety**: Prolonged training can generate significant heat and stress on GPU hardware. Users should ensure their systems are adequately cooled and configured for sustained high-load operations.

## 4. No Warranty

- This software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.