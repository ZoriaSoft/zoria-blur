; Zoria Blur - Windows installer (Inno Setup 6).
; Built by packaging\build_windows.ps1 (installer step), which passes /DAppVersion=<pubspec version> and
; /DSourceDir=<dist\ZoriaBlur>. Manual build from the repo root:
;   ISCC.exe /DAppVersion=0.1.0+1 packaging\installer.iss
; Per-user by default (no UAC prompt): %LOCALAPPDATA%\Programs\Zoria Blur. The "install for all users"
; dialog / the /ALLUSERS switch moves it to Program Files (needs admin).
; Uninstall removes the program; the user's logs/projects in %LOCALAPPDATA%\ZoriaBlur stay unless the user
; says yes to the question at the end of an interactive uninstall (silent uninstalls always keep them).

; Flavor: "" = the light installer (no NVIDIA engine), "-NVIDIA" = with engine-cuda (build_windows.ps1 makes both)
#ifndef Flavor
  #define Flavor ""
#endif
#ifndef SourceDir
  #define SourceDir "..\dist\ZoriaBlur"
#endif
#ifndef AppVersion
  ; fallback: the version stamped into ZoriaBlur.exe by flutter (e.g. 0.1.0+1)
  #define AppVersion GetStringFileInfo(SourceDir + "\ZoriaBlur.exe", "ProductVersion")
#endif
; 0.1.0+1 -> short "0.1.0" (file name, display) and numeric "0.1.0.1" (file version resource)
#define PlusPos Pos("+", AppVersion)
#if PlusPos > 0
  #define ShortVersion Copy(AppVersion, 1, PlusPos - 1)
  #define BuildNo Copy(AppVersion, PlusPos + 1)
#else
  #define ShortVersion AppVersion
  #define BuildNo "0"
#endif
#define NumVersion ShortVersion + "." + BuildNo

[Setup]
AppId={{6F1B2C4E-8A3D-4E57-9B21-5C7D0A9E3F14}
AppName=Zoria Blur
AppVersion={#ShortVersion}
AppVerName=Zoria Blur {#ShortVersion}
AppPublisher=ZoriaSoft
AppPublisherURL=https://zoriasoft.com
AppCopyright=Copyright (C) ZoriaSoft
VersionInfoVersion={#NumVersion}
VersionInfoCompany=ZoriaSoft
VersionInfoProductName=Zoria Blur
VersionInfoDescription=Zoria Blur Setup
; per-user install without admin; the user may still pick "all users" (dialog or /ALLUSERS)
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog commandline
DefaultDirName={autopf}\Zoria Blur
DefaultGroupName=Zoria Blur
DisableProgramGroupPage=yes
DisableDirPage=auto
DisableReadyPage=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
WizardStyle=modern
SetupIconFile=..\ui\windows\runner\resources\app_icon.ico
UninstallDisplayIcon={app}\ZoriaBlur.exe
UninstallDisplayName=Zoria Blur
Compression=lzma2/max
SolidCompression=yes
OutputDir=..\dist
OutputBaseFilename=ZoriaBlur-Setup-{#ShortVersion}{#Flavor}
; close a running Zoria Blur / engine before replacing or removing its files
CloseApplications=yes
CloseApplicationsFilter=*.exe,*.dll
RestartApplications=no
; Turkish is the default (first entry, no UI-language detection); English is one pick away
ShowLanguageDialog=yes
LanguageDetectionMethod=none

[Languages]
Name: "tr"; MessagesFile: "compiler:Languages\Turkish.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
tr.RemoveUserData=Zoria Blur kaldırıldı.%n%nKayıtlar ve proje dosyaları da silinsin mi?%n%1%n%n(Hayır derseniz bu klasör yerinde kalır.)
en.RemoveUserData=Zoria Blur has been removed.%n%nAlso delete its logs and project files?%n%1%n%n(Choose No to keep this folder.)

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[InstallDelete]
; an update replaces the bundled engines / flutter data wholesale, so files dropped in a newer build do not linger
; (engine-cuda too: an install without it must not leave an old CUDA engine that the app would try first)
Type: filesandordirs; Name: "{app}\engine"
Type: filesandordirs; Name: "{app}\engine-cuda"
Type: filesandordirs; Name: "{app}\data"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Zoria Blur"; Filename: "{app}\ZoriaBlur.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\Zoria Blur"; Filename: "{app}\ZoriaBlur.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\ZoriaBlur.exe"; Description: "{cm:LaunchProgram,Zoria Blur}"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; anything the program left inside its own install folders (the user's data lives elsewhere, see below)
Type: filesandordirs; Name: "{app}\engine"
Type: filesandordirs; Name: "{app}\engine-cuda"
Type: filesandordirs; Name: "{app}\data"
Type: dirifempty; Name: "{app}"

[Code]
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
begin
  if (CurUninstallStep = usPostUninstall) and (not UninstallSilent) then
  begin
    DataDir := ExpandConstant('{localappdata}\ZoriaBlur');
    if DirExists(DataDir) then
      if MsgBox(FmtMessage(CustomMessage('RemoveUserData'), [DataDir]), mbConfirmation,
                MB_YESNO or MB_DEFBUTTON2) = IDYES then
        DelTree(DataDir, True, True, True);
  end;
end;
