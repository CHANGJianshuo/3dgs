import shutil
p = "/root/autodl-tmp/gsplat_repo/examples/simple_trainer.py"
src = open(p).read()
if "compress_level=1" in src:
    print("already patched")
    raise SystemExit
shutil.copyfile(p, p + ".pre_compress_bak")
old = (
    'imageio.imwrite(\n'
    '                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",\n'
    '                    canvas,\n'
    '                )'
)
new = (
    'imageio.imwrite(\n'
    '                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",\n'
    '                    canvas,\n'
    '                    compress_level=1,\n'
    '                )'
)
assert old in src, "could not find imwrite block"
src = src.replace(old, new)
open(p, "w").write(src)
print("patched OK")
