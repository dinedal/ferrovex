const { spawnSync } = require('node:child_process')
const path = require('node:path')

const skipPrepare = process.env.FERROVEX_SKIP_PREPARE === '1' || process.env.CI === 'true'

if (skipPrepare) {
  console.log('[prepare] skipping native build in CI/bootstrap publish')
  process.exit(0)
}

const buildScript = path.join(__dirname, 'build-native.js')
const result = spawnSync(process.execPath, [buildScript, '--release'], { stdio: 'inherit' })

if (result.error) {
  console.error(result.error.message)
  process.exit(1)
}

process.exit(result.status ?? 1)
