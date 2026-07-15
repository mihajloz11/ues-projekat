module.exports = {
    flowFile: "flows.json",
    credentialSecret: "wifi-csi-tinyml-local",
    uiPort: process.env.PORT || 1880,
    httpAdminRoot: "/red",
    httpNodeRoot: "/",
    diagnostics: { enabled: false, ui: false },
    runtimeState: { enabled: false, ui: false },
    logging: {
        console: {
            level: "info",
            metrics: false,
            audit: false
        }
    },
    editorTheme: {
        projects: { enabled: false }
    },
    functionExternalModules: false
};
