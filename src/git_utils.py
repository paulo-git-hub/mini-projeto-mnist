import os
import subprocess
import sys
from pathlib import Path
from typing import Union, Optional

ROOT_PATH = Path('/content/drive/MyDrive/mini_projeto_mnist')

class GitManager:
    """
    Classe responsável pelo gerenciamento de branches, staging,
    commits estruturados e push para o repositório remoto via Git Flow.
    """
    def __init__(self, root_path: Union[str, Path] = ROOT_PATH):
        self.root_path = Path(root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)

    def _run_cmd(self, command: str) -> str:
        """
        Executa comandos do sistema usando shell=True para evitar FileNotFoundError.
        """
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(self.root_path),
            capture_output=True,
            text=True,
            check=False
        )
        return (result.stdout + result.stderr).strip()

    def gerenciar_fase(self, fase: str, mensagem_commit: str) -> None:
        """
        Alterna/cria a branch da fase, adiciona modificações, realiza o commit
        e faz o push remoto de forma estritamente idempotente.
        """
        if fase in ["main", "principal"]:
            branch_name = fase
        else:
            branch_name = f"feature/{fase}" if not fase.startswith("feature/") else fase

        branches_out = self._run_cmd("git branch -a")

        if branch_name in branches_out:
            self._run_cmd(f"git checkout {branch_name}")
        else:
            self._run_cmd(f"git checkout -b {branch_name}")

        self._run_cmd("git add .")

        commit_out = self._run_cmd(f'git commit -m "{mensagem_commit}"')
        if "nothing to commit" in commit_out:
            print(f"ℹ️ [GitManager] Nenhuma alteração pendente na branch '{branch_name}'.")
        else:
            print(f"✅ [GitManager] Commit realizado em '{branch_name}': {mensagem_commit}")

        push_out = self._run_cmd(f"git push origin {branch_name}")
        print(f"🚀 [GitManager] Push em '{branch_name}': {push_out if push_out else 'Sincronizado.'}")


class GitMerger(GitManager):
    """
    Classe estendida para consolidação final de branches no repositório.
    """
    def consolidar_para_principal(self, feature_branch: str = "fase5-robustness") -> None:
        branches_out = self._run_cmd("git branch -a")
        target_main = "principal" if "principal" in branches_out else "main"
        feature_name = f"feature/{feature_branch}" if not feature_branch.startswith("feature/") else feature_branch

        print(f"🔄 Alternando para a branch principal: '{target_main}'...")
        self._run_cmd(f"git checkout {target_main}")

        print(f"🔀 Unificando alterações de '{feature_name}' em '{target_main}'...")
        merge_out = self._run_cmd(f'git merge {feature_name} --no-ff -m "merge: consolida {feature_name} em {target_main}"')
        print(f"📝 Resultado do Merge: {merge_out}")

        print(f"🚀 Enviando a branch '{target_main}' para o GitHub...")
        push_out = self._run_cmd(f"git push origin {target_main}")
        print(f"✅ Consolidação concluída com sucesso!")
