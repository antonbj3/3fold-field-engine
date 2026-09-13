PYTHON ?= python3

.PHONY: verify-csg
verify-csg:
	$(PYTHON) scripts/verify_csg.py

# SDK paths and architecture are explicit environment inputs to the builder.
.PHONY: build-prepared-service verify-prepared-cpu verify-prepared-gpu
build-prepared-service:
	$(PYTHON) scripts/build_prepared_service.py --output "$(BUILD_DIR)"
verify-prepared-cpu:
	$(PYTHON) scripts/verify_prepared_service.py --output "$(EVIDENCE_DIR)"
verify-prepared-gpu:
	$(PYTHON) scripts/verify_prepared_service.py --gpu --build "$(BUILD_DIR)" --plate "$(PLATE_STL)" --output "$(EVIDENCE_DIR)" $(VERIFY_ARGS)

.PHONY: verify verify-declared verify-inventory
verify:
	$(PYTHON) scripts/verify.py $(VERIFY_ARGS)
verify-declared:
	$(PYTHON) scripts/verify.py --declared $(VERIFY_ARGS)
verify-inventory:
	$(PYTHON) scripts/verify.py --inventory $(VERIFY_ARGS)

.PHONY: verify-geometry
verify-geometry:
	$(PYTHON) scripts/verify.py --declared --group geometry_cpu $(VERIFY_ARGS)

.PHONY: verify-recipe
verify-recipe:
	$(PYTHON) scripts/verify.py --declared --group recipe_cpu $(VERIFY_ARGS)

.PHONY: verify-canonical-adaptive
verify-canonical-adaptive:
	$(PYTHON) scripts/verify_geometry.py --module field_adaptive_canonical_probe.py --module field_adaptive_input_audit.py --output "$(EVIDENCE_DIR)"
