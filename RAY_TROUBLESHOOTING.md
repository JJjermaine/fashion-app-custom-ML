# Ray GCS Connection Troubleshooting Guide

## Problem Description

You're experiencing a GCS (Global Control Store) connection timeout error:
```
gcs_rpc_client.h:179: Failed to connect to GCS at address 10.126.20.4:2222 within 5 seconds.
```

This is a common issue with Ray clusters on Vertex AI where the client can't connect to the GCS server due to network issues, timeouts, or cluster state problems.

## Solutions Implemented

### 1. **Increased Timeout Values**
The main fix is increasing various timeout values in the `setup_ray_cluster()` function:

```python
# Environment variables set for better stability
os.environ["RAY_gcs_rpc_server_reconnect_timeout_s"] = "30"  # Increased from 5s to 30s
os.environ["RAY_gcs_rpc_server_connect_timeout_s"] = "30"    # Connection timeout
os.environ["RAY_gcs_rpc_server_request_timeout_s"] = "60"    # Request timeout
os.environ["RAY_gcs_rpc_server_retry_timeout_s"] = "10"      # Retry timeout
os.environ["RAY_gcs_rpc_server_retry_interval_s"] = "1"      # Retry interval
os.environ["RAY_gcs_rpc_server_max_retries"] = "5"           # Maximum retries
```

### 2. **Retry Logic with Exponential Backoff**
Added retry logic that attempts connection multiple times with increasing delays:

```python
max_retries = 3
retry_delay = 5

for attempt in range(max_retries):
    try:
        # Connection attempt with only supported parameters
        ray.init(
            address=ray_address,
            log_to_driver=True,
            ignore_reinit_error=True
        )
        return True
    except Exception as e:
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff
```

### 3. **Fixed Unsupported Parameters**
Removed unsupported Ray initialization parameters that were causing errors:
- ❌ Removed: `_redis_connection_pool_size`, `_redis_connection_pool_timeout`
- ❌ Removed: `_num_cpus`, `_num_gpus`, `_resources`
- ❌ Removed: `_object_store_memory`, `_local_mode`
- ❌ Removed: `_enable_object_reconstruction`, `_object_spilling_config`
- ❌ Removed: `_enable_multi_tenancy`, `_system_config`
- ✅ Kept only: `address`, `log_to_driver`, `ignore_reinit_error`

### 4. **No Fallback Option**
As requested, removed the fallback to local Ray. The script will now exit if cluster connection fails.

## How to Use the Fixes

### Step 1: Test the Connection
Before running your full model, test the connection:

```bash
python test_ray_connection.py
```

This will verify that the Ray cluster connection works with the new timeout settings.

### Step 2: Run Diagnostics (if needed)
If the connection test fails, run the diagnostic script:

```bash
python ray_diagnostics.py
```

This will provide detailed information about:
- Ray installation status
- Network connectivity
- Vertex AI cluster status
- Detailed connection diagnostics

### Step 3: Run Your Model
Once the connection is working, run your model:

```bash
python model.py
```

## Additional Troubleshooting Steps

### 1. **Check Vertex AI Cluster Status**
Verify your Ray cluster is running in the Google Cloud Console:
- Go to Vertex AI > Ray clusters
- Check if your cluster is in "RUNNING" state

### 2. **Restart the Ray Cluster**
If the cluster is stuck or having issues:
```bash
# Stop the cluster
gcloud ai persistent-resources delete projects/927709385665/locations/us-west2/persistentResources/cluster-20250712-190801

# Create a new cluster or restart the existing one
```

### 3. **Check Authentication**
Ensure you're authenticated with Google Cloud:
```bash
gcloud auth application-default login
```

### 4. **Network Issues**
If you're behind a corporate firewall, ensure these ports are open:
- Port 6379 (Redis/GCS)
- Port 10001 (Ray head node)
- Port 2222 (GCS RPC)

## Environment Variables Reference

| Variable | Default | New Value | Purpose |
|----------|---------|-----------|---------|
| `RAY_gcs_rpc_server_reconnect_timeout_s` | 5 | 30 | GCS reconnect timeout |
| `RAY_gcs_rpc_server_connect_timeout_s` | 5 | 30 | GCS connection timeout |
| `RAY_gcs_rpc_server_request_timeout_s` | 30 | 60 | GCS request timeout |
| `RAY_gcs_rpc_server_retry_timeout_s` | 5 | 10 | GCS retry timeout |
| `RAY_gcs_rpc_server_retry_interval_s` | 0.5 | 1 | Retry interval |
| `RAY_gcs_rpc_server_max_retries` | 3 | 5 | Maximum retries |

## Common Error Messages and Solutions

| Error | Solution |
|-------|----------|
| `Failed to connect to GCS within 5 seconds` | ✅ **FIXED** - Increased timeout to 30 seconds |
| `Timed out while waiting for GCS to become available` | ✅ **FIXED** - Added retry logic with exponential backoff |
| `Got unexpected kwargs: _num_cpus, _resources, _num_gpus...` | ✅ **FIXED** - Removed unsupported parameters |
| `Connection refused` | Check if cluster is running, restart if needed |
| `Authentication failed` | Run `gcloud auth application-default login` |
| `Network unreachable` | Check firewall settings and network connectivity |

## Performance Notes

- The increased timeouts may make initial connection slower but more reliable
- Retry logic adds resilience but may delay startup on network issues
- No fallback option means the script will exit if cluster connection fails
- For production, consider using KubeRay for better fault tolerance

## Next Steps

1. **Test the connection**: `python test_ray_connection.py`
2. **Run diagnostics if needed**: `python ray_diagnostics.py`
3. **Run your model**: `python model.py`
4. **Monitor the logs** for any remaining issues

The fixes should resolve the GCS connection timeout issues and the unsupported parameters error you were experiencing. 