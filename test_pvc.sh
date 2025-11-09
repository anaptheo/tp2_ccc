kubectl run pvc-check \
  --rm -i --tty \
  --image=busybox \
  --restart=Never \
  --namespace=anatheobald \
  --overrides='
{
  "spec": {
    "volumes": [
      {
        "name": "model-volume",
        "persistentVolumeClaim": {
          "claimName": "project2-pvc-anatheobald"
        }
      }
    ],
    "containers": [
      {
        "name": "check",
        "image": "busybox",
        "command": ["ls", "-l", "/model"],
        "volumeMounts": [
          {
            "mountPath": "/model",
            "name": "model-volume"
          }
        ]
      }
    ]
  }
}
'
