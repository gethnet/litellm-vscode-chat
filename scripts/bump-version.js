#!/usr/bin/env node
/* eslint-disable no-undef */
/* eslint-disable @typescript-eslint/no-require-imports */

const fs = require("fs");
const path = require("path");

const packageJsonPath = path.join(__dirname, "..", "package.json");
const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, "utf8"));

// Get the version bump type from command line args (patch, minor, major, dev)
const bumpType = process.argv[2] || "patch";

if (!["patch", "minor", "major", "dev"].includes(bumpType)) {
	console.error("Usage: npm run bump-version [patch|minor|major|dev]");
	console.error("  patch: 0.1.0 -> 0.1.1");
	console.error("  minor: 0.1.0 -> 0.2.0");
	console.error("  major: 0.1.0 -> 1.0.0");
	console.error("  dev:   0.1.0 -> 0.1.0-dev (or keeps -dev if present)");
	process.exit(1);
}

// Parse current version
const versionMatch = packageJson.version.match(/^(\d+)\.(\d+)\.(\d+)(-dev)?$/);
if (!versionMatch) {
	console.error(`Invalid version format in package.json: ${packageJson.version}`);
	process.exit(1);
}

const major = parseInt(versionMatch[1]);
const minor = parseInt(versionMatch[2]);
const patch = parseInt(versionMatch[3]);
const isDev = !!versionMatch[4];

// Calculate new version
let newVersion;
switch (bumpType) {
	case "major":
		newVersion = `${major + 1}.0.0`;
		break;
	case "minor":
		newVersion = `${major}.${minor + 1}.0`;
		break;
	case "patch":
		newVersion = `${major}.${minor}.${patch + 1}`;
		break;
	case "dev":
		newVersion = isDev ? packageJson.version : `${major}.${minor}.${patch}-dev`;
		break;
	default:
		newVersion = `${major}.${minor}.${patch + 1}`;
		break;
}

// Update package.json
packageJson.version = newVersion;
fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, "\t") + "\n");
console.log(`Version bumped: ${packageJson.version} -> ${newVersion}`);

// Return the new version for use in scripts
process.stdout.write(`${packageJson.version} -> ${newVersion}`);
