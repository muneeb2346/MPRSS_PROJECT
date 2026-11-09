/*
 * MPRSS Module 1 - Behavioral Hook Agent (Frida)
 *
 * Hooks sensitive Android framework APIs commonly abused by malware and
 * reports each invocation as a structured, timestamped event. Categories
 * chosen to mirror what the proposal's Module 1 description calls out
 * (contact access, internet connection, data transmission) plus the
 * broader sensitive-API surface malware typically touches.
 *
 * Each event is sent back to the Python harness via send() as JSON:
 *   { ts, category, api, detail, pid }
 *
 * This does NOT block or modify behavior - every hook calls through to
 * the real implementation via .call()/.apply() after logging, so the
 * app runs completely normally. This is passive observation only.
 */

import Java from 'frida-java-bridge';

console.log('[behavior_hooks] Script loaded, checking for Java...');

function safeStr(x) {
    try {
        if (x === null || x === undefined) return String(x);
        return x.toString();
    } catch (e) {
        return '<unstringifiable>';
    }
}

function report(category, api, detail) {
    send({
        ts: Date.now(),
        category: category,
        api: api,
        detail: detail,
        pid: Process.id
    });
}

var waitAttempts = 0;
function waitForJava(callback) {
    waitAttempts++;
    if (waitAttempts % 20 === 0) {
        console.log('[behavior_hooks] Still waiting for Java... attempt ' + waitAttempts +
            ' (Java defined: ' + (typeof Java !== 'undefined') + ')');
    }
    if (typeof Java !== 'undefined' && Java.available) {
        console.log('[behavior_hooks] Java is now available!');
        callback();
    } else {
        setTimeout(function () { waitForJava(callback); }, 100);
    }
}

waitForJava(function () {
    try {
Java.perform(function () {

    // ------------------------------------------------------------------
    // Permission checks
    // ------------------------------------------------------------------
    try {
        var ContextImpl = Java.use('android.app.ContextImpl');
        ContextImpl.checkSelfPermission.overload('java.lang.String').implementation = function (perm) {
            report('permission', 'Context.checkSelfPermission', perm);
            return this.checkSelfPermission(perm);
        };
    } catch (e) { console.log('[hook skip] checkSelfPermission: ' + e); }

    // ------------------------------------------------------------------
    // SMS
    // ------------------------------------------------------------------
    try {
        var SmsManager = Java.use('android.telephony.SmsManager');
        SmsManager.sendTextMessage.overload(
            'java.lang.String', 'java.lang.String', 'java.lang.String',
            'android.app.PendingIntent', 'android.app.PendingIntent'
        ).implementation = function (dest, sc, text, sentIntent, deliveryIntent) {
            report('sms', 'SmsManager.sendTextMessage', 'to=' + safeStr(dest));
            return this.sendTextMessage(dest, sc, text, sentIntent, deliveryIntent);
        };
    } catch (e) { console.log('[hook skip] SmsManager.sendTextMessage: ' + e); }

    // ------------------------------------------------------------------
    // Contacts / SMS / Call-log access via ContentResolver.query
    // (covers contacts, sms, call log - all use the same query() entrypoint,
    // distinguished by the content:// URI they pass in)
    // ------------------------------------------------------------------
    try {
        var ContentResolver = Java.use('android.content.ContentResolver');
        ContentResolver.query.overload(
            'android.net.Uri', '[Ljava.lang.String;', 'java.lang.String',
            '[Ljava.lang.String;', 'java.lang.String'
        ).implementation = function (uri, projection, selection, selArgs, sortOrder) {
            var uriStr = safeStr(uri);
            var kind = 'content';
            if (uriStr.indexOf('contacts') !== -1) kind = 'contacts';
            else if (uriStr.indexOf('sms') !== -1) kind = 'sms';
            else if (uriStr.indexOf('call_log') !== -1 || uriStr.indexOf('calls') !== -1) kind = 'call_log';
            report('content_query', 'ContentResolver.query[' + kind + ']', uriStr);
            return this.query(uri, projection, selection, selArgs, sortOrder);
        };
    } catch (e) { console.log('[hook skip] ContentResolver.query: ' + e); }

    // ------------------------------------------------------------------
    // Camera
    // ------------------------------------------------------------------
    try {
        var Camera = Java.use('android.hardware.Camera');
        Camera.open.overload('int').implementation = function (id) {
            report('camera', 'Camera.open', 'id=' + id);
            return this.open(id);
        };
    } catch (e) { console.log('[hook skip] Camera.open: ' + e); }

    try {
        var CameraManager = Java.use('android.hardware.camera2.CameraManager');
        CameraManager.openCamera.overload(
            'java.lang.String', 'android.hardware.camera2.CameraDevice$StateCallback', 'android.os.Handler'
        ).implementation = function (id, cb, handler) {
            report('camera', 'CameraManager.openCamera', 'id=' + safeStr(id));
            return this.openCamera(id, cb, handler);
        };
    } catch (e) { console.log('[hook skip] CameraManager.openCamera: ' + e); }

    // ------------------------------------------------------------------
    // Location
    // ------------------------------------------------------------------
    try {
        var LocationManager = Java.use('android.location.LocationManager');
        LocationManager.getLastKnownLocation.overload('java.lang.String').implementation = function (provider) {
            report('location', 'LocationManager.getLastKnownLocation', 'provider=' + safeStr(provider));
            return this.getLastKnownLocation(provider);
        };
    } catch (e) { console.log('[hook skip] getLastKnownLocation: ' + e); }

    // ------------------------------------------------------------------
    // Telephony / device identifiers (matches Drebin-style feature names
    // used in Module 2's training schema, e.g. TelephonyManager.getSimCountryIso)
    // ------------------------------------------------------------------
    try {
        var TelephonyManager = Java.use('android.telephony.TelephonyManager');
        ['getDeviceId', 'getSubscriberId', 'getSimSerialNumber', 'getLine1Number', 'getSimCountryIso'].forEach(function (m) {
            try {
                TelephonyManager[m].overload().implementation = function () {
                    var result = this[m]();
                    report('telephony', 'TelephonyManager.' + m, safeStr(result));
                    return result;
                };
            } catch (e) { console.log('[hook skip] TelephonyManager.' + m + ': ' + e); }
        });
    } catch (e) { console.log('[hook skip] TelephonyManager class: ' + e); }

    // ------------------------------------------------------------------
    // Network - basic URL/socket connections (data transmission)
    // ------------------------------------------------------------------
    try {
        var URL = Java.use('java.net.URL');
        URL.openConnection.overload().implementation = function () {
            report('network', 'URL.openConnection', safeStr(this.toString()));
            return this.openConnection();
        };
    } catch (e) { console.log('[hook skip] URL.openConnection: ' + e); }

    // ------------------------------------------------------------------
    // File I/O (writes only - reads are extremely noisy and low-signal)
    // ------------------------------------------------------------------
    try {
        var FileOutputStream = Java.use('java.io.FileOutputStream');
        FileOutputStream.$init.overload('java.lang.String').implementation = function (path) {
            report('file_io', 'FileOutputStream.<init>', 'path=' + safeStr(path));
            return this.$init(path);
        };
    } catch (e) { console.log('[hook skip] FileOutputStream: ' + e); }

    // ------------------------------------------------------------------
    // Command execution (shell command signatures - matches Drebin's
    // "Commands signature" category from Module 2)
    // ------------------------------------------------------------------
    try {
        var Runtime = Java.use('java.lang.Runtime');
        Runtime.exec.overload('java.lang.String').implementation = function (cmd) {
            report('command_exec', 'Runtime.exec', safeStr(cmd));
            return this.exec(cmd);
        };
    } catch (e) { console.log('[hook skip] Runtime.exec: ' + e); }


    // Self-test: directly invoke a hooked method to prove the pipeline works,
// independent of whether the target app organically triggers it.

    console.log('[behavior_hooks] All hooks installed. Monitoring...');
});
} catch (outerErr) {
    console.log('[behavior_hooks] FATAL: Java.perform itself threw: ' + outerErr + '\n' + outerErr.stack);
}
});