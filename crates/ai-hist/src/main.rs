//! Internal thin binary wrapper. All logic lives in `ai_hist_engine`; the
//! production Node CLI calls the public TypeScript SDK instead.

fn main() -> anyhow::Result<()> {
    ai_hist_engine::run()
}
