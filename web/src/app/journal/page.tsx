import type { Metadata } from "next";
import { Source_Serif_4 } from "next/font/google";
import Link from "next/link";
import { Fragment } from "react";

import { JOURNAL, type JournalEntry, STATUS } from "@/content/journal";

const serif = Source_Serif_4({ subsets: ["latin"], variable: "--font-serif", display: "swap" });

export const metadata: Metadata = {
  title: "What we did · grpo-lab",
  description: "The grpo-lab lab notebook: each milestone, what it found, and what went wrong.",
};

/** Backticked spans become inline code; everything else stays plain text. */
function Prose({ text }: { text: string }) {
  return text.split(/(`[^`]+`)/).map((part, i) =>
    part.startsWith("`") && part.endsWith("`") ? (
      <code key={i}>{part.slice(1, -1)}</code>
    ) : (
      <Fragment key={i}>{part}</Fragment>
    ),
  );
}

const dateFmt = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "long", year: "numeric" });

function Entry({ entry }: { entry: JournalEntry }) {
  return (
    <article id={entry.id} className="journal-entry" aria-labelledby={`${entry.id}-title`}>
      <p className="journal-meta">
        <time dateTime={entry.date}>{dateFmt.format(new Date(`${entry.date}T12:00:00Z`))}</time>
        <span aria-hidden> · </span>
        {entry.milestone}
        {entry.status === "in-progress" && <span className="journal-live">In progress</span>}
      </p>
      <h2 id={`${entry.id}-title`} className="journal-title">
        <a href={`#${entry.id}`}>{entry.title}</a>
      </h2>
      {entry.figures && (
        <dl className="journal-figures">
          {entry.figures.map((f) => (
            <div key={f.label}>
              <dt>
                <Prose text={f.label} />
              </dt>
              <dd>{f.value}</dd>
            </div>
          ))}
        </dl>
      )}
      <div className="journal-body">
        {entry.body.map((p, i) => (
          <p key={i}>
            <Prose text={p} />
          </p>
        ))}
        {entry.caveat && (
          <p className="journal-caveat">
            <em>Caveat.</em> <Prose text={entry.caveat} />
          </p>
        )}
      </div>
      {entry.links && (
        <ul className="journal-links" aria-label="Links">
          {entry.links.map((l) => (
            <li key={l.href}>
              {l.href.startsWith("/") ? (
                <Link href={l.href}>{l.label} →</Link>
              ) : (
                <a href={l.href}>{l.label} ↗</a>
              )}
            </li>
          ))}
        </ul>
      )}
    </article>
  );
}

export default function JournalPage() {
  const entries = [...JOURNAL].reverse(); // newest first; the source file reads oldest first
  return (
    <div className={`journal ${serif.variable} flex-1`}>
      <main className="mx-auto w-full max-w-5xl px-4 pt-12 pb-24 sm:px-6">
        <header className="journal-head">
          <h1>From-scratch GRPO lifts a 0.5B model six points on GSM8K. Here is how we got there.</h1>
          <p className="journal-lede">
            A lab notebook for grpo-lab: one entry per milestone, newest first. Each title states
            what we found, every figure is measured, and each caveat says what went wrong or what
            the number doesn&rsquo;t show.
          </p>
          <ul className="journal-status" aria-label="Current state">
            {STATUS.map((s) => (
              <li key={s}>{s}</li>
            ))}
          </ul>
        </header>
        {entries.map((e) => (
          <Entry key={e.id} entry={e} />
        ))}
      </main>
    </div>
  );
}
