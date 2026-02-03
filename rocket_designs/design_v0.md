# Rocket Design v0 - Simple Cylinder

## Screenshot
![Design v0](screenshots/design_v0.png)

## Overview
Initial minimal rocket design for proof-of-concept testing. Simple cylinder body with no visual details.

## Physical Specifications

| Property | Value |
|----------|-------|
| Body shape | Cylinder |
| Radius | 0.1 m |
| Height | 2.0 m (size parameter 1.0 = half-height) |
| Mass | 10 kg |
| Starting height | 50 m |

## Actuators

| Name | Type | Gear | Range | Purpose |
|------|------|------|-------|---------|
| thrust_x | Motor | 25 | [-1, 1] | Lateral X control |
| thrust_y | Motor | 25 | [-1, 1] | Lateral Y control |
| thrust_z | Motor | 200 | [0, 1] | Main vertical thrust |

## Observations
- Very basic cylinder shape - not representative of real rocket
- No landing legs - cannot actually "land" properly
- No visual distinction between top/bottom of rocket
- Single thruster site at bottom

## Limitations
- Unrealistic shape makes it hard to judge orientation visually
- No landing gear means success is just "hovering near ground"
- 2D landing behavior difficult to enforce without constraints

## XML File
`env/xml_files/single_rocket_test.xml`
