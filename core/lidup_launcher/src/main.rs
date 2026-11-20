#![windows_subsystem = "windows"]

use std::path::{Path, PathBuf};
use std::process::Command;

#[cfg(windows)]
fn show_error_dialog(title: &str, message: &str) {
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "user32")]
    extern "system" {
        fn MessageBoxW(hwnd: isize, text: *const u16, caption: *const u16, utype: u32) -> i32;
    }
    let to_wide = |s: &str| -> Vec<u16> {
        std::ffi::OsStr::new(s)
            .encode_wide()
            .chain(std::iter::once(0))
            .collect()
    };
    let t = to_wide(title);
    let m = to_wide(message);
    unsafe {
        MessageBoxW(0, m.as_ptr(), t.as_ptr(), 0x10);
    }
}

#[cfg(not(windows))]
fn show_error_dialog(_title: &str, _message: &str) {}

fn find_workspace_and_pythonw(start_dir: &Path) -> Option<(PathBuf, PathBuf)> {
    let mut cur = Some(start_dir);
    for _ in 0..5 {
        if let Some(dir) = cur {
            let pyw = dir.join(".venv").join("Scripts").join("pythonw.exe");
            if pyw.exists() {
                return Some((dir.to_path_buf(), pyw));
            }
            cur = dir.parent();
        }
    }
    None
}

fn main() {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let found = find_workspace_and_pythonw(&exe_dir)
        .or_else(|| std::env::current_dir().ok().and_then(|cwd| find_workspace_and_pythonw(&cwd)));

    let Some((workspace, pythonw)) = found else {
        show_error_dialog(
            "LidUp Studio",
            "Não foi possível localizar .venv\\Scripts\\pythonw.exe na pasta do projeto.",
        );
        return;
    };

    #[allow(unused_mut)]
    let mut cmd = Command::new(&pythonw);
    cmd.args(["-m", "lidup.studio"])
        .arg(&workspace)
        .current_dir(&workspace);

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    match cmd.status() {
        Ok(status) => {
            if !status.success() {
                if let Some(code) = status.code() {
                    show_error_dialog(
                        "LidUp Studio — Encerrado com erro",
                        &format!("O aplicativo encerrou com código {code}."),
                    );
                }
            }
        }
        Err(err) => {
            show_error_dialog(
                "LidUp Studio — Erro ao iniciar",
                &format!("Falha ao executar {}: {}", pythonw.display(), err),
            );
        }
    }
}
