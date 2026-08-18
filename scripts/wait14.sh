#!/usr/bin/env bash
until [ -f /root/afterimage/store_14b/manifest.json ]; do sleep 20; done
echo "COMPRESSION_DONE"
python3 -c "
import json
m=json.load(open(\"/root/afterimage/store_14b/manifest.json\"))
print(\"ORIGINAL  : %.3f GB\" % (m[\"total_orig_bytes\"]/1e9))
print(\"COMPRESSED: %.3f GB\" % (m[\"total_comp_bytes\"]/1e9))
print(\"RATIO     : %.3fx\" % m[\"ratio\"])
"
