#!/bin/bash
# Script to help debug VM identification in Storm cluster
# Usage: ./debug_vm_identification.sh

echo "═══════════════════════════════════════════════════════"
echo "    Storm VM Identification Debugger"
echo "═══════════════════════════════════════════════════════"
echo ""

# Check if Storm is installed
if ! command -v storm &> /dev/null; then
    echo "❌ Error: 'storm' command not found. Is Storm installed?"
    exit 1
fi

echo "1️⃣  Checking local VM hostname information..."
echo "   Short hostname:  $(hostname)"
echo "   FQDN:            $(hostname -f 2>/dev/null || echo 'N/A')"
echo "   IP Address:      $(hostname -I | awk '{print $1}')"
echo ""

echo "2️⃣  Listing registered Storm supervisors..."
echo ""
storm list-supervisors 2>/dev/null || {
    echo "❌ Could not connect to Nimbus. Is Storm running?"
    echo "   Check:"
    echo "   - Is Nimbus running? (ps aux | grep nimbus)"
    echo "   - Is storm.yaml configured correctly?"
    exit 1
}
echo ""

echo "3️⃣  Checking placement CSV file..."
echo ""

# Try to find the CSV file
CSV_FILE=""
if [ -n "$STORM_HOME" ] && [ -f "$STORM_HOME/conf/component-placement.csv" ]; then
    CSV_FILE="$STORM_HOME/conf/component-placement.csv"
elif [ -f "/etc/storm/component-placement.csv" ]; then
    CSV_FILE="/etc/storm/component-placement.csv"
elif [ -f "component-placement.csv" ]; then
    CSV_FILE="component-placement.csv"
fi

if [ -z "$CSV_FILE" ]; then
    echo "⚠️  Warning: Placement CSV not found in standard locations:"
    echo "   - \$STORM_HOME/conf/component-placement.csv"
    echo "   - /etc/storm/component-placement.csv"
    echo "   - ./component-placement.csv"
else
    echo "✅ Found placement CSV: $CSV_FILE"
    echo ""
    echo "Contents:"
    cat "$CSV_FILE"
    echo ""

    # Check if hostnames in CSV match registered supervisors
    echo "4️⃣  Verifying hostname matches..."
    echo ""
    while IFS=, read -r component host; do
        # Skip header and comments
        if [[ "$component" == "Component" ]] || [[ "$component" == component* ]] || [[ "$component" == \#* ]]; then
            continue
        fi

        if [ -n "$host" ]; then
            host=$(echo "$host" | xargs)  # trim whitespace

            # Check exact match
            if storm list-supervisors 2>/dev/null | grep -q "$host"; then
                echo "✅ '$host' found in registered supervisors"
            else
                echo "❌ '$host' NOT found in registered supervisors"
                echo "   → Check if hostname matches supervisor hostname"
                echo "   → Try using: $(hostname) or $(hostname -f)"
            fi
        fi
    done < "$CSV_FILE"
fi

echo ""
echo "5️⃣  Checking Nimbus logs for scheduling decisions..."
echo ""

# Try to find nimbus log
NIMBUS_LOG=""
if [ -n "$STORM_HOME" ] && [ -f "$STORM_HOME/logs/nimbus.log" ]; then
    NIMBUS_LOG="$STORM_HOME/logs/nimbus.log"
elif [ -f "/var/log/storm/nimbus.log" ]; then
    NIMBUS_LOG="/var/log/storm/nimbus.log"
elif [ -f "logs/nimbus.log" ]; then
    NIMBUS_LOG="logs/nimbus.log"
fi

if [ -z "$NIMBUS_LOG" ]; then
    echo "⚠️  Warning: Nimbus log not found"
else
    echo "Recent CsvOneToOneScheduler activity:"
    echo ""
    tail -n 50 "$NIMBUS_LOG" | grep -i "CsvOneToOne\|Available Supervisors\|Resolved" || echo "No recent scheduler activity"
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo "Troubleshooting Tips:"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "If components are not scheduling correctly:"
echo ""
echo "1. Ensure hostnames in CSV match supervisor hostnames:"
echo "   → Use: storm list-supervisors"
echo "   → Update CSV with exact hostnames"
echo ""
echo "2. Or override supervisor hostname in storm.yaml:"
echo "   supervisor.hostname: \"vm1\""
echo ""
echo "3. Check Nimbus logs for detailed scheduling decisions:"
echo "   tail -f $NIMBUS_LOG | grep CsvOneToOne"
echo ""
echo "4. Verify supervisors have available worker slots:"
echo "   → Add more ports in storm.yaml:"
echo "   supervisor.slots.ports:"
echo "     - 6700"
echo "     - 6701"
echo ""
echo "For more details, see: docs/VM_IDENTIFICATION.md"
echo ""

