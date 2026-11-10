----------------------------- MODULE Program -----------------------------
EXTENDS Naturals
VARIABLES phase, verified, published
vars == <<phase, verified, published>>
Init == /\ phase = "Ready" /\ verified = FALSE /\ published = FALSE
Run == /\ phase = "Ready" /\ phase' = "Running" /\ UNCHANGED <<verified, published>>
Finish == /\ phase = "Running" /\ phase' = "Staging" /\ UNCHANGED <<verified, published>>
Verify == /\ phase = "Staging" /\ phase' = "Verified" /\ verified' = TRUE /\ UNCHANGED published
Publish == /\ phase = "Verified" /\ phase' = "Published" /\ published' = TRUE /\ UNCHANGED verified
Fail == /\ phase \in {"Running", "Staging", "Verified"} /\ phase' = "Failed" /\ UNCHANGED <<verified, published>>
Interrupt == /\ phase \in {"Running", "Staging", "Verified"} /\ phase' = "Interrupted" /\ UNCHANGED <<verified, published>>
Done == /\ phase \in {"Published", "Failed", "Interrupted"} /\ UNCHANGED vars
Next == Run \/ Finish \/ Verify \/ Publish \/ Fail \/ Interrupt \/ Done
PublishedOnlyVerified == published => verified
FailedNeverPublished == phase \in {"Failed", "Interrupted"} => ~published
PublicationIsTerminal == published => phase = "Published"
=============================================================================
