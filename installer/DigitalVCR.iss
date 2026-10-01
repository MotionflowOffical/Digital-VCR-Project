#define MyAppName "Digital VCR"
#define MyAppVersion "8.1"
#define MyAppPublisher "MotionflowOffical"
#define MyAppURL "https://github.com/MotionflowOffical/Digital-VCR-Project"
#define MyAppExeName "DigitalVCR.exe"

[Setup]
AppId={{D31A01E8-E3F6-4E0B-879A-6B6E20BAF98C}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} V{#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\Digital VCR
DefaultGroupName=Digital VCR
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
OutputDir=output
OutputBaseFilename=Digital-VCR-V8.1-Setup
SetupIconFile=..\assets\DigitalVCR.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
VersionInfoVersion=8.1.0.0
VersionInfoCompany={#MyAppPublisher}
VersionInfoDescription=Digital VCR V8.1 Installer
VersionInfoProductName={#MyAppName}
VersionInfoProductVersion={#MyAppVersion}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
Source: "..\dist\DigitalVCR\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Digital VCR"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\Digital VCR"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch Digital VCR"; Flags: nowait postinstall skipifsilent
