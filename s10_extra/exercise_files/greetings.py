import click

@click.group()
def cli():
    pass

@cli.command()
@click.option('--count', default=1, help='Number of greetings.')
@click.option('--name', prompt='Your name', help='The person to greet.')
def hello(count, name):
    """Simple program that greets NAME for a total of COUNT times."""
    for x in range(count):
        click.echo(f"Hello {name}!")

@cli.command()
@click.option('--count', default=1, help='Number of greetings.')
@click.option('--name', prompt='Your name', help='The person to greet.')
def howdy(count, name):
    """Another greeting program that says Howdy."""
    for x in range(count):
        click.echo(f"Howdy {name}!")

if __name__ == '__main__':
    cli()

