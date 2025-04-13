from typing import Optional

import click


class AliasedGroup(click.Group):
    """
    Group class that allows commands to be aliased.

    http://click.pocoo.org/5/advanced/
    """

    def get_command(self, ctx: click.Context, cmd_name: str) -> Optional[click.Command]:
        """Get a command by its name or alias."""
        rv = click.Group.get_command(self, ctx, cmd_name)
        if rv is not None:
            return rv
        matches = [x for x in self.list_commands(ctx) if x.startswith(cmd_name)]
        if not matches:
            return None
        if len(matches) == 1:
            return click.Group.get_command(self, ctx, matches[0])
        return ctx.fail(f"Too many matches: {', '.join(sorted(matches))}")

    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str, click.Command, list[str]]:
        """Resolve a command and its arguments."""
        # always return the full command name
        _, cmd, args = super().resolve_command(ctx, args)
        if cmd is None or cmd.name is None:
            raise click.BadParameter(
                f"Command '{args[0]}' not found. Use 'moro --help' for a list of commands."
            )
        return cmd.name, cmd, args
