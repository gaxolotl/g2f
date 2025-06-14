#define MyAppName "g2f GIF Splitter"
#define MyAppVersion "0.4"
#define MyAppPublisher "Anomic Forums Community"
#define MyAppURL "https://github.com/gaxolotl/g2f"
#define MyAppExeName "main.exe"
#define MyAppAssocName MyAppName + " File"
#define MyAppAssocExt ".myp"
#define MyAppAssocKey StringChange(MyAppAssocName, " ", "") + MyAppAssocExt

[Setup]
AppId={{B04CDC75-79ED-4746-84B7-2BEF9E7D4EE9}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={localappdata}\{#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExeName}
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
ChangesAssociations=yes
DisableProgramGroupPage=yes
LicenseFile=LICENSE
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=setupout
OutputBaseFilename=g2fsetup
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "dist\main\main.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\main\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs

[Registry]
Root: HKCU; Subkey: "Software\Classes\{#MyAppAssocExt}\OpenWithProgids"; ValueType: string; ValueName: "{#MyAppAssocKey}"; ValueData: ""; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Classes\{#MyAppAssocKey}"; ValueType: string; ValueName: ""; ValueData: "{#MyAppAssocName}"; Flags: uninsdeletekey
Root: HKCU; Subkey: "Software\Classes\{#MyAppAssocKey}\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExeName},0"
Root: HKCU; Subkey: "Software\Classes\{#MyAppAssocKey}\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""

[Icons]
Name: "{userprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{userdesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
