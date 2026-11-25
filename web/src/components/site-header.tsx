import Link from "next/link";

export function SiteHeader() {
  return (
    <header className="border-b">
      <div className="mx-auto flex h-12 w-full max-w-6xl items-center gap-6 px-4">
        <Link href="/" className="font-semibold tracking-tight">
          grpo-lab
        </Link>
        <nav className="flex gap-4 text-sm text-muted-foreground">
          <Link href="/" className="hover:text-foreground">
            Runs
          </Link>
        </nav>
        <a
          href="https://github.com/Padraigobrien08/grpo-lab"
          className="ml-auto text-sm text-muted-foreground hover:text-foreground"
        >
          GitHub
        </a>
      </div>
    </header>
  );
}
