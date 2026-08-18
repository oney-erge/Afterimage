#!/usr/bin/env bash
ls -la /root/afterimage/store_14b/manifest.json 2>/dev/null && python3 -c "
import json
m=json.load(open(\"/root/afterimage/store_14b/manifest.json\"))
print(\"tensors:\", len(m[\"tensors\"]))
print(\"ORIGINAL  : %.3f GB\" % (m[\"total_orig_bytes\"]/1e9))
print(\"COMPRESSED: %.3f GB\" % (m[\"total_comp_bytes\"]/1e9))
print(\"RATIO     : %.3fx\" % m[\"ratio\"])
" || echo "manifest not ready yet"
du -sh /root/afterimage/store_14b 2>/dev/null
