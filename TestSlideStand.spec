# -*- mode: python ; coding: utf-8 -*-


block_cipher = None

from PyInstaller.utils.hooks import collect_submodules
hidden_zaber = collect_submodules('zaber_motion')
hidden_zaber2 = collect_submodules('zaber_motion_bindings_windows')

a = Analysis(
    ['src\\TestSlideStand\\__main__.py'],
    pathex=[],
    binaries=[(r'venv\Lib\site-packages\zaber_motion_bindings_windows\zaber-motion-lib-windows-amd64.dll', 'zaber_motion_bindings_windows')],
    datas=[],
    hiddenimports=hidden_zaber+hidden_zaber2+['skimage.filters.thresholding', 'matplotlib.backends.backend_pdf', 'TestSlideStand.mplwidget'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='TestSlideStand',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='__main__',
)
