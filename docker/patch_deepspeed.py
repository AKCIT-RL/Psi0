import pathlib

p = pathlib.Path('/workspace/.venv/lib/python3.10/site-packages/deepspeed/git_version_info.py')
txt = p.read_text()
old = 'for op_name, builder in ALL_OPS.items():\n    op_compatible = builder.is_compatible()'
new = (
    'for op_name, builder in ALL_OPS.items():\n'
    '    try:\n'
    '        op_compatible = builder.is_compatible()\n'
    '    except Exception:\n'
    '        op_compatible = False'
)
if old in txt:
    p.write_text(txt.replace(old, new))
    print('deepspeed patched OK')
else:
    print('deepspeed already patched or pattern not found — skipping')
