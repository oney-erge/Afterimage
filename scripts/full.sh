python3 -c "
import json
r=json.load(open(\"/root/afterimage/results/headtohead_14b.json\"))
m=r[\"manifest\"]
print(\"=== MANIFEST ===\")
for k,v in m.items(): print(\"  %s: %s\" % (k,v))
print()
print(\"=== OUR RUN (full record) ===\")
for res in r[\"results\"]:
    for k,v in res.items():
        if k==\"token_ids\": v=str(v)
        print(\"  %-24s %s\" % (k,v))
"
echo "--- store on disk ---"
du -sb /root/afterimage/store_14b | cut -f1 | xargs -I{} python3 -c "print(\"  store_14b actual bytes on disk: %.3f GB\" % ({}/1e9))"
echo "--- original checkpoint on disk ---"
du -sb ~/.cache/huggingface/hub/models--Qwen--Qwen3-14B | cut -f1 | xargs -I{} python3 -c "print(\"  hf cache (incl. split files): %.3f GB\" % ({}/1e9))"
