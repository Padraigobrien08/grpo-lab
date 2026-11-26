import Link from "next/link";

export function SiteHeader() {
  return (
    <header className="border-b">
      <div className="mx-auto flex h-12 w-full max-w-6xl items-center gap-4 px-4 whitespace-nowrap sm:gap-6">
        <Link href="/" className="font-semibold tracking-tight">
          grpo-lab
        </Link>
        <nav className="flex gap-3 text-sm text-muted-foreground sm:gap-4">
          <Link href="/" className="hover:text-foreground">
            Runs
          </Link>
          <Link href="/compare" className="hover:text-foreground">
            Compare
          </Link>
          <Link href="/journal" className="hover:text-foreground">
            What we did
          </Link>
        </nav>
        <a
          href="https://github.com/Padraigobrien08/grpo-lab"
          // Below sm the nav needs the room; every journal entry links to GitHub anyway.
          className="ml-auto hidden text-sm text-muted-foreground hover:text-foreground sm:inline"
        >
          GitHub
        </a>
      </div>
    </header>
  );
}
