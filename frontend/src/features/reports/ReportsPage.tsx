import { useEffect, useState } from "react";
import { InspectionReportsApi, apiErrorMessage } from "../../services/api";
import { useWorkstation } from "../../workstation";
import type { InspectionReport } from "../../types";

/*
 * Saved inspection reports (backend app/api/inspection_reports.py): the list,
 * and one report with its findings, evidence frame, the inspector's note and
 * disposition. The findings are a record and can't be edited; "Export PDF"
 * prints the report (the browser's Save as PDF).
 */

const STATUS_TEXT: Record<string, string> = {
  NOT_INSPECTED: "NOT INSPECTED",
  PRESENT: "PRESENT",
  MISSING_CANDIDATE: "CHECKING",
  MISSING_CONFIRMED: "MISSING",
  UNCERTAIN: "UNCERTAIN",
  OCCLUDED: "OCCLUDED",
};

const when = (iso: string | undefined) => (iso ? new Date(iso).toLocaleString() : "—");

interface Props {
  openReportId: string | null;
  onOpenReport: (id: string | null) => void;
}

export function ReportsPage({ openReportId, onOpenReport }: Props) {
  const [reports, setReports] = useState<InspectionReport[]>([]);
  const [report, setReport] = useState<InspectionReport | null>(null);
  const [note, setNote] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const workstation = useWorkstation();

  const refresh = () =>
    InspectionReportsApi.list()
      .then((list) => {
        setReports(list);
        if (!openReportId && list[0]) onOpenReport(list[0].id);
      })
      .catch((err) => setError(apiErrorMessage(err)));
  useEffect(() => {
    void refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    setReport(null);
    if (!openReportId) return;
    InspectionReportsApi.get(openReportId)
      .then((r) => {
        setReport(r);
        setNote(r.note);
      })
      .catch((err) => setError(apiErrorMessage(err)));
  }, [openReportId]);

  const review = async (disposition?: "approved" | "rework") => {
    if (!report) return;
    setBusy(true);
    try {
      const updated = await InspectionReportsApi.review(report.id, {
        note,
        disposition,
        reviewer: workstation.operator || null,
      });
      setReport(updated);
      void refresh();
    } catch (err) {
      setError(apiErrorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="page reports-page">
      <div className="page-head no-print">
        <div>
          <div className="section-label">Reports</div>
          <h2 className="page-title">Inspection records</h2>
        </div>
      </div>
      {error && <p className="error no-print">{error}</p>}

      <div className="reports-layout">
        <aside className="card report-list no-print" aria-label="Saved reports">
          {reports.length === 0 && (
            <p className="hint">No reports yet. Run an inspection on the Live AR screen and use "Save report".</p>
          )}
          {reports.map((r) => (
            <button
              key={r.id}
              className={`report-item${r.id === openReportId ? " active" : ""}`}
              onClick={() => onOpenReport(r.id)}
              aria-pressed={r.id === openReportId}
            >
              <span className="head">
                <span className={`verdict-mini ${r.verdict}`}>{r.verdict}</span>
                <span className="mono hint">{when(r.created_at)}</span>
              </span>
              <span className="meta">
                {r.model_name}
                {r.station ? ` · ${r.station}` : ""}
              </span>
              {r.disposition && <span className={`tag ${r.disposition === "approved" ? "ok" : "warn"}`}>{r.disposition}</span>}
            </button>
          ))}
        </aside>

        {report && (
          <article className="report" aria-label="Inspection report">
            <div className="report-head">
              <div>
                <div className="section-label">Inspection record · {report.id}</div>
                <h2 className="page-title">{report.model_name}</h2>
                <div className="hint">
                  {report.station || "[no station]"} · {report.operator || "[no operator]"} · {when(report.created_at)} · run{" "}
                  {report.run_id} · {report.passes} passes
                </div>
              </div>
              <span className={`verdict-badge ${report.verdict}`}>{report.verdict}</span>
            </div>

            <div className="count-grid wide">
              <div>
                <strong>{report.summary.required}</strong>Required
              </div>
              <div className="ok">
                <strong>{report.summary.present}</strong>Verified present
              </div>
              <div className="bad">
                <strong>{report.summary.missing}</strong>Confirmed missing
              </div>
              <div>
                <strong>{report.summary.uncertain + report.summary.not_inspected + report.summary.candidates}</strong>Not decided
              </div>
            </div>

            <div className="report-body">
              <section className="card" aria-label="Findings">
                <h3>Findings</h3>
                <div className="table-scroll">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Component</th>
                        <th>Status</th>
                        <th>Score</th>
                        <th>Valid views</th>
                        <th>Evidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(report.components ?? []).map((c) => (
                        <tr key={c.id} className={c.status === "MISSING_CONFIRMED" ? "row-bad" : ""}>
                          <td>
                            <div className="mono strong">{c.id}</div>
                            <div className="hint">
                              {c.display_name}
                              {!c.required ? " · optional" : ""}
                            </div>
                          </td>
                          <td>
                            <span className={`status-tag ${c.status}`}>{STATUS_TEXT[c.status] ?? c.status}</span>
                          </td>
                          <td className="mono">{c.score != null ? `${(c.score * 100).toFixed(0)}%` : "—"}</td>
                          <td className="mono">{c.valid_observations}</td>
                          <td className="hint">{c.reason}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {report.notes.map((n) => (
                  <p key={n} className="hint">
                    ⚠ {n}
                  </p>
                ))}
              </section>

              <div className="report-side">
              <section className="card" aria-label="Evidence frame">
                <h3>Evidence</h3>
                {report.evidence ? (
                  <a href={InspectionReportsApi.evidenceUrl(report.id)} target="_blank" rel="noreferrer" title="Open full size">
                    <img
                      className="evidence"
                      src={InspectionReportsApi.evidenceUrl(report.id)}
                      alt="Camera frame at the time of the report, with each component's outline in its status colour"
                    />
                  </a>
                ) : (
                  <p className="hint">No camera frame was available when this report was saved.</p>
                )}
              </section>
              <section className="card sign-off" aria-label="Sign-off">
                <label className="field grow">
                  Inspector note
                  <input value={note} onChange={(e) => setNote(e.target.value)} placeholder="e.g. cap found on the bench, refitted" />
                </label>
                <div className="button-row no-print">
                  <button onClick={() => void review()} disabled={busy || note === report.note}>
                    Save note
                  </button>
                  <button onClick={() => window.print()}>Export PDF</button>
                  <button className="danger" onClick={() => void review("rework")} disabled={busy}>
                    Send to rework
                  </button>
                  <button className="primary" onClick={() => void review("approved")} disabled={busy || report.verdict !== "PASS"} title={report.verdict !== "PASS" ? "Only a PASS can be approved" : undefined}>
                    Approve
                  </button>
                </div>
                {report.disposition && (
                  <p className="hint">
                    {report.disposition === "approved" ? "Approved" : "Sent to rework"}
                    {report.disposition_by ? ` by ${report.disposition_by}` : ""} · {when(report.disposition_at)}
                  </p>
                )}
              </section>
              </div>
            </div>

          </article>
        )}
      </div>
    </div>
  );
}
