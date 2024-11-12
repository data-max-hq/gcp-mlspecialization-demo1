## Setting Up Google Cloud Storage, VM, and Initializing the Script for GCP ML Specialization Demo

This guide provides instructions to create a Google Cloud Storage bucket with a simulated directory structure, set up a VM with full API access, and initialize a script from a GitHub repository, all within Cloud Shell.

---

### 1. Creating a Google Cloud Storage Bucket

In Cloud Shell, use the following command to create a Google Cloud Storage bucket:

```bash
gsutil mb -l <REGION> gs://<YOUR_BUCKET_NAME>
```

* Replace `<REGION>` with the desired region for your bucket (e.g., `us-central1`).
* Replace `<YOUR_BUCKET_NAME>` with a globally unique name.

**Example:**

```bash
gsutil mb -l us-central1 gs://dataset_bucket_demo1
```

#### 1.1 Creating "Directories" (Using Object Prefixes)

Simulate directories by uploading a dummy file to your bucket. This creates the specified directory structure:

```bash
touch dummy.txt

gsutil cp dummy.txt gs://<YOUR_BUCKET_NAME>/pipeline_module/taxi_chicago_pipeline/
gsutil cp dummy.txt gs://<YOUR_BUCKET_NAME>/pipeline_root/taxi_chicago_pipeline/

rm dummy.txt
```

* Replace `<YOUR_BUCKET_NAME>` with the name of your bucket.

**Example:**

```bash
touch dummy.txt

gsutil cp dummy.txt gs://dataset_bucket_demo1/pipeline_module/taxi_chicago_pipeline/
gsutil cp dummy.txt gs://dataset_bucket_demo1/pipeline_root/taxi_chicago_pipeline/

rm dummy.txt
```

---

### 2. Creating a VM with Full API Access

To create a VM with full API access:

```bash
gcloud compute instances create <VM_NAME> \
    --zone=<ZONE> \
    --machine-type=e2-medium \
    --image-family=debian-12 \
    --image-project=debian-cloud \
    --boot-disk-size=20GB \
    --scopes=cloud-platform
```

* Replace `<VM_NAME>` with the desired name for your VM.
* Replace `<ZONE>` with the desired zone.

**Example:**

```bash
gcloud compute instances create my-debian-vm \
    --zone=us-central1-a \
    --machine-type=e2-medium \
    --image-family=debian-12 \
    --image-project=debian-cloud \
    --boot-disk-size=20GB \
    --scopes=cloud-platform
```

---

### 3. Script Initialization Guide

This section covers steps to initialize the script for the GCP ML Specialization Demo, setting up the environment by cloning and running a repository script.

#### Prerequisites

Ensure `sudo` privileges are enabled on your system.

#### Steps

1. **Install Git**

   Ensure Git is installed:

   ```bash
   sudo apt update
   sudo apt install -y git
   ```

2. **Clone the GitHub Repository**

   Clone the repository for the GCP ML Specialization Demo:

   ```bash
   git clone https://github.com/data-max-hq/gcp-mlspecialization-demo1.git
   ```

3. **Run the Initialization Script**

   Navigate to the cloned repository and run the startup script:

   ```bash
   cd gcp-mlspecialization-demo1
   chmod +x startup.sh
   sudo ./startup.sh
   ```

---

### Key Considerations and Best Practices

* **Service Account**: For better security, use a service account with limited permissions instead of the `cloud-platform` scope in production environments.
* **Firewall Rules**: Configure firewall rules according to your application requirements.
* **Region and Zone Selection**: Choose these strategically based on latency, availability, and cost needs.
* **Cleanup**: Remember to delete the VM and bucket after finishing to avoid unnecessary charges.

Following these steps will set up your GCP environment and initialize the GCP ML Specialization Demo. If you encounter any issues, please refer to the repository's documentation.
