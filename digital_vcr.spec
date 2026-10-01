# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules, collect_dynamic_libs, collect_data_files
from pathlib import Path

block_cipher = None

hiddenimports = []
hiddenimports += collect_submodules('vcr')
hiddenimports += collect_submodules('PIL')
hiddenimports += collect_submodules('customtkinter')
hiddenimports += collect_submodules('moderngl')
hiddenimports += collect_submodules('glfw')
hiddenimports += [
    'tkinter',
    'tkinter.ttk',
    'tkinter.filedialog',
    'tkinter.messagebox',
    'cv2',
    'numpy',
    'imageio_ffmpeg',
    'moderngl',
    'glfw',
]

datas = []
datas += collect_data_files('imageio_ffmpeg')
datas += collect_data_files('customtkinter')
assets_dir = Path('assets')
if assets_dir.exists():
    datas.append((str(assets_dir / 'DigitalVCR.ico'), 'assets'))
    datas.append((str(assets_dir / 'DigitalVCR.png'), 'assets'))

binaries = []
binaries += collect_dynamic_libs('cv2')
binaries += collect_dynamic_libs('moderngl')
binaries += collect_dynamic_libs('glfw')
native_dll = Path('vcr/native/digital_vcr_core.dll')
if native_dll.exists():
    binaries.append((str(native_dll), 'vcr/native'))

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
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
    name='DigitalVCR',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    icon='assets/DigitalVCR.ico',
    version='version_info.txt',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DigitalVCR',
)
